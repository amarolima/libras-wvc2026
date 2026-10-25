"""
Ablação em escada da ResNet50 (Tabela II do artigo).

Cada degrau acrescenta um fator ao anterior:
  C0   congelado,    sem embaralhamento, sem aumento, 10 épocas, LR 1e-4  (protocolo de referência)
  C1   congelado,    com embaralhamento, sem aumento, 10 épocas, LR 1e-4
  C2   congelado,    com embaralhamento, com aumento, 30 épocas, LR 1e-4  (mesmo orçamento de C3)
  C2b  congelado,    com embaralhamento, com aumento, 30 épocas, LR 1e-5  (controle da taxa de aprendizado)
  C3   descongelado, com embaralhamento, com aumento, 30 épocas, LR 1e-5  (fine-tuning de conv5_*)

Cada treino roda em um processo próprio: encadear dezenas de treinos da
ResNet50 no mesmo processo vaza memória até o sistema derrubar o processo.
O notebook chama este script uma vez por (cenário, degrau, semente).

Uso:
  python experimentos/resnet50_escada.py CENARIO DEGRAU SEMENTE \
      --dataset DIR_LSWH100 --saida DIR_RESULTADOS

Saídas (em DIR_RESULTADOS):
  ablacao_escada_treino.json                   acurácia, macro-F1, épocas e duração (incremental)
  predicoes/pred_{CENARIO}_{DEGRAU}_s{SEMENTE}.npy  predição por imagem de teste (para o McNemar)
  predicoes/y_{CENARIO}.npy                    rótulos de teste na mesma ordem
"""
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

# Cresce a memória da GPU sob demanda, em vez de reservar tudo de uma vez.
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

CLASSES_5 = ['S15a', 'S203', 'S19a', 'S19c', 'S185']
CLASSES_6 = CLASSES_5 + ['S177']
CLASSES_10 = CLASSES_5 + ['S1ce', 'S1de', 'S1a3', 'S10e', 'S192']
CENARIOS = {'5_classes': CLASSES_5, '6_classes': CLASSES_6, '10_classes': CLASSES_10}

TAMANHO_LOTE = 16

ESCADA = {
    'C0_original':   {'backbone': 'congelado',    'shuffle': False, 'aug': False, 'epocas': 10, 'paciencia': 3, 'lr': 1e-4},
    'C1_shuffle':    {'backbone': 'congelado',    'shuffle': True,  'aug': False, 'epocas': 10, 'paciencia': 3, 'lr': 1e-4},
    'C2_protocolo':  {'backbone': 'congelado',    'shuffle': True,  'aug': True,  'epocas': 30, 'paciencia': 6, 'lr': 1e-4},
    'C2b_lr_baixo':  {'backbone': 'congelado',    'shuffle': True,  'aug': True,  'epocas': 30, 'paciencia': 6, 'lr': 1e-5},
    'C3_finetuning': {'backbone': 'descongelado', 'shuffle': True,  'aug': True,  'epocas': 30, 'paciencia': 6, 'lr': 1e-5},
}


def carregar_dados_coco(dir_dataset, split, classes):
    """Caminhos e rótulos de um split, na ordem das anotações COCO.

    As anotações do LSWH100 vêm agrupadas por classe; é por isso que treinar
    sem embaralhamento (C0) produz lotes com apenas duas classes.
    """
    with open(os.path.join(dir_dataset, f'{split}_annotations.coco.json')) as f:
        dados = json.load(f)
    cat_id_para_codigo = {c['id']: c['name'] for c in dados['categories']}
    img_por_id = {img['id']: img for img in dados['images']}
    caminhos, rotulos = [], []
    for anotacao in dados['annotations']:
        codigo = cat_id_para_codigo.get(anotacao['category_id'])
        if codigo in classes:
            info = img_por_id.get(anotacao['image_id'])
            if info:
                caminhos.append(os.path.join(dir_dataset, info['file_name']))
                rotulos.append(classes.index(codigo))
    return caminhos, rotulos


def montar_dataset(caminhos, rotulos, cfg, treino):
    # Aumento de dados leve, aplicado somente ao treino (Seção III-D).
    aumentador = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.1),
    ], name="aumentador")

    def carregar(caminho, rotulo):
        img = tf.io.read_file(caminho)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        return img, rotulo

    ds = tf.data.Dataset.from_tensor_slices((caminhos, rotulos))
    if treino and cfg['shuffle']:
        ds = ds.shuffle(buffer_size=len(caminhos))
    ds = ds.map(carregar, num_parallel_calls=tf.data.AUTOTUNE)
    if treino and cfg['aug']:
        ds = ds.map(lambda x, y: (aumentador(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(TAMANHO_LOTE).prefetch(tf.data.AUTOTUNE)


def construir_modelo(n_classes, cfg):
    """ResNet50 (ImageNet) + GAP + Densa(256, ReLU) + dropout 0,5 + softmax."""
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    if cfg['backbone'] == 'descongelado':
        # Descongela apenas o último bloco residual (conv5_*, 32 das 175 camadas).
        base.trainable = True
        corte = next(i for i, c in enumerate(base.layers) if c.name.startswith('conv5_'))
        for camada in base.layers[:corte]:
            camada.trainable = False
    else:
        base.trainable = False

    modelo = models.Sequential([
        base, layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'), layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax')
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['lr']),
                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo


def treinar(cenario, degrau, semente, dir_dataset, dir_saida):
    classes = CENARIOS[cenario]
    cfg = ESCADA[degrau]
    dir_pred = os.path.join(dir_saida, 'predicoes')
    os.makedirs(dir_pred, exist_ok=True)

    tf.keras.utils.set_random_seed(semente)
    treino = carregar_dados_coco(dir_dataset, 'train', classes)
    val = carregar_dados_coco(dir_dataset, 'val', classes)
    teste = carregar_dados_coco(dir_dataset, 'test', classes)
    y_teste = np.array(teste[1])
    np.save(os.path.join(dir_pred, f"y_{cenario}.npy"), y_teste)

    modelo = construir_modelo(len(classes), cfg)
    parada = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg['paciencia'],
                                              restore_best_weights=True)
    t0 = time.time()
    hist = modelo.fit(montar_dataset(*treino, cfg, treino=True),
                      validation_data=montar_dataset(*val, cfg, treino=False),
                      epochs=cfg['epocas'], callbacks=[parada], verbose=0)
    duracao = time.time() - t0

    pred = np.argmax(modelo.predict(montar_dataset(*teste, cfg, treino=False), verbose=0), axis=1)
    np.save(os.path.join(dir_pred, f"pred_{cenario}_{degrau}_s{semente}.npy"), pred)
    rel = classification_report(y_teste, pred, zero_division=0, output_dict=True)
    r = {
        'semente': semente,
        'epocas_rodadas': len(hist.history['loss']),
        'duracao_segundos': duracao,
        'acuracia_teste': float(np.mean(pred == y_teste)),
        'macro_f1_teste': rel['macro avg']['f1-score'],
    }

    # Acrescenta ao JSON, substituindo um resultado anterior da mesma semente.
    caminho_json = os.path.join(dir_saida, 'ablacao_escada_treino.json')
    resultados = {}
    if os.path.exists(caminho_json):
        with open(caminho_json) as f:
            resultados = json.load(f)
    lista = resultados.setdefault(cenario, {}).setdefault(degrau, [])
    lista = [x for x in lista if x['semente'] != semente] + [r]
    resultados[cenario][degrau] = sorted(lista, key=lambda x: x['semente'])
    with open(caminho_json, 'w') as f:
        json.dump(resultados, f, indent=2)

    print(f"[{cenario}/{degrau}/s{semente}] acc={r['acuracia_teste']:.4f} "
          f"F1={r['macro_f1_teste']:.4f} {r['epocas_rodadas']} épocas {duracao:.0f}s", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cenario', choices=list(CENARIOS))
    ap.add_argument('degrau', choices=list(ESCADA))
    ap.add_argument('semente', type=int)
    ap.add_argument('--dataset', required=True, help='diretório do LSWH100 (com os *_annotations.coco.json)')
    ap.add_argument('--saida', required=True, help='diretório de resultados')
    a = ap.parse_args()
    treinar(a.cenario, a.degrau, a.semente, a.dataset, a.saida)
