# Duas armadilhas de avaliação no reconhecimento de configurações de mão em Libras

Código e resultados do artigo

> L. A. C. S. O. Miguel, J. N. Gois, W. L. Passos, A. R. L. Zachi, L. F. Almeida, A. A. Lima, G. M. Araujo.
> **Two Evaluation Pitfalls in Landmark- and Pixel-Based Hand Configuration Recognition for Brazilian Sign Language.**
> XXI Workshop de Visão Computacional (WVC 2026), Campinas, 2026.

*English summary: this repository reproduces every table and figure of the paper above. It compares a landmark
branch (MediaPipe + MLP) with a pixel branch (ResNet50) on the synthetic LSWH100 dataset and shows two evaluation
pitfalls that favor landmarks: accuracy computed only over detected hands, and a frozen CNN backbone. The notebook
and comments are in Portuguese.*

## O que o artigo mostra

1. **Acurácia condicionada à detecção.** Corrigir o pré-processamento entregue ao MediaPipe eleva a detecção de
   0,770 para 0,880 (5 classes), mas faz a acurácia medida só nas mãos detectadas **cair** de 0,98 para 0,94,
   enquanto a acurácia fim-a-fim sobe de 0,755 para 0,828.
2. **Backbone congelado.** Com o protocolo de treino igualado, descongelar o último bloco da ResNet50 vale de 13 a
   26 pontos percentuais; outros 4 a 7 vêm de diferenças de protocolo raramente reportadas.

## Conteúdo

| Caminho | Conteúdo |
|---|---|
| [`libras_wvc2026.ipynb`](libras_wvc2026.ipynb) | notebook comentado com todos os experimentos; gera as Tabelas I–IV e as Figs. 2 e 3 |
| [`libras_wvc2026.py`](libras_wvc2026.py) | o mesmo notebook em texto (formato *percent* do jupytext), para revisar diferenças no git |
| [`experimentos/resnet50_escada.py`](experimentos/resnet50_escada.py) | treino da ResNet50 na ablação em escada C0–C3 (um treino por processo) |
| [`resultados/`](resultados/) | resultados usados no artigo (JSON), pontos-chave extraídos e predições por imagem |
| [`figuras/`](figuras/) | figuras geradas pelo notebook |

### Onde está cada resultado do artigo

| Artigo | Seção do notebook | Arquivo em `resultados/` |
|---|---|---|
| Fig. 2 (detecção por condição e por orientação) | 2 | `ablacao_preprocessamento.json` |
| Tabela I (efeito de seleção) | 4 | `mlp.json`, `landmarks_condA.npz`, `landmarks_condH.npz` |
| Representação dos landmarks (Seção IV-B) | 4 | `mlp.json` |
| Tabela II (escada C0–C3) | 5 | `ablacao_escada_treino.json`, `predicoes/` |
| Testes de McNemar | 6 | `mcnemar.json` |
| Tabela III (comparação fim-a-fim) | 7 | `mlp.json`, `ablacao_escada_treino.json` |
| Fig. 3 (matrizes de confusão) | 8 | `predicoes/` |
| Tabela IV (latência) | 9 | `latencia_final.json` |
| Versões do MediaPipe (Seção IV-E) | 10 | `comparacao_versoes_mediapipe.json` |

`multiplas_sementes_mlp.json` e `mlp_condicao_otimizada.json` são as saídas originais dos MLPs; o notebook retreina os
MLPs e confere que reproduz esses valores.

## Conjunto de dados

O artigo usa o **LSWH100** (*Libras SignWriting Handshape dataset*), que é público e **não está incluído** neste
repositório.

* Download: <https://zenodo.org/records/10635628>
* Referência: V. C. Lobo-Neto e H. Pedrini, "LSWH100: A handshape dataset for Brazilian sign language (Libras) using
  SignWriting", *Data in Brief*, v. 56, 110780, 2024. DOI [10.1016/j.dib.2024.110780](https://doi.org/10.1016/j.dib.2024.110780).

Depois de baixar e descompactar, o diretório do conjunto deve ter esta estrutura:

```
LSWH100/
├── train_annotations.coco.json
├── val_annotations.coco.json
├── test_annotations.coco.json
├── class_list.csv
├── front/   back/   left/   right/      # imagens RGB 500x500, uma pasta por orientação
```

Para o código encontrar o conjunto, faça uma das duas coisas:

```bash
# opção 1: variável de ambiente (recomendada)
export LSWH100_DIR=/caminho/para/LSWH100

# opção 2: coloque (ou crie um link para) o conjunto em dados/LSWH100 dentro deste repositório
mkdir -p dados && ln -s /caminho/para/LSWH100 dados/LSWH100
```

Só os experimentos que partem das imagens precisam do conjunto. Sem ele, o notebook roda inteiro a partir de
`resultados/`.

## Como reproduzir

```bash
conda create -n libras-wvc2026 python=3.12 -y
conda activate libras-wvc2026
pip install -r requirements.txt
jupyter lab libras_wvc2026.ipynb
```

Por padrão o notebook usa os resultados salvos e retreina apenas os MLPs, que levam cerca de 4 minutos. Para refazer um
experimento a partir das imagens, ligue a chave correspondente na célula de configuração:

| Chave | Experimento | Tempo aproximado* |
|---|---|---|
| `EXECUTAR_DETECCAO` | ablação de pré-processamento do MediaPipe (8 condições) | 2 min |
| `EXECUTAR_LANDMARKS` | extração dos pontos-chave nas condições A e H | 5 min |
| `EXECUTAR_RESNET` | escada C0–C3 da ResNet50, 75 treinos | 10 h |
| `EXECUTAR_LATENCIA` | latência de inferência | 1 min |
| `EXECUTAR_VERSOES_MP` | detecção com a API `tasks` do MediaPipe | 1 min |

\* NVIDIA RTX 4050 Laptop (6 GB).

Observações:

* **MediaPipe 0.10.21.** O pipeline de referência usa a API `mp.solutions.hands`, removida no MediaPipe 1.0.0. Para
  medir a versão 1.0.1 (Seção 10 do notebook), use um ambiente separado com `mediapipe==1.0.1`; a API legada é pulada
  automaticamente.
* **Determinismo.** Na máquina do artigo, a detecção, os pontos-chave e os MLPs refeitos do zero reproduzem
  exatamente os valores publicados. Os treinos da ResNet50 usam convoluções cuDNN não determinísticas: repetir um
  treino pode mudar uma ou duas predições (no teste, C0 / 5 classes / semente 0 deu 0,820 contra 0,815 publicado),
  diferença dentro do desvio entre sementes da Tabela II. Em outro hardware ou outra versão do TensorFlow podem
  aparecer diferenças na terceira casa decimal.
* **Latência** depende do hardware; os números salvos são da máquina do artigo.
* Para rodar um único treino da ResNet50 fora do notebook:
  `python experimentos/resnet50_escada.py 5_classes C3_finetuning 0 --dataset $LSWH100_DIR --saida resultados`

## Como citar

```bibtex
@inproceedings{miguel2026pitfalls,
  author    = {Leonardo A. C. S. O. Miguel and Jonathan N. Gois and Wesley L. Passos and Alessandro R. L. Zachi
               and Luciana F. Almeida and Amaro A. Lima and Gabriel M. Araujo},
  title     = {Two Evaluation Pitfalls in Landmark- and Pixel-Based Hand Configuration Recognition
               for {Brazilian} Sign Language},
  booktitle = {Anais do XXI Workshop de Vis{\~a}o Computacional (WVC)},
  year      = {2026}
}
```

## Licença

Código sob a licença MIT (ver [`LICENSE`](LICENSE)). O LSWH100 segue a licença definida pelos seus autores no Zenodo.
