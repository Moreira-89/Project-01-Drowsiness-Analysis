# Projeto de Visão Computacional utilizando OpenCV <br>

Nosso objetivo é construir um detector de sonolência utilizando Mediapipe e o OpenCV. Faremos isso através da análise de coordenadas faciais em um vídeo de captura do rosto. Basicamente, a ideia geral do projeto é criar uma solução inteligente para motoristas de carro e caminhão. Sabemos que as pessoas podem ficar cansadas e esgotadas fisicamente após dirigir por muito tempo, e por lei, não é permitido dirigir nesse estado de esgotamento físico. É necessário estar bem de saúde para evitar acidentes. Portanto, nosso objetivo é criar um algoritmo para detectar o estado de sonolência em motoristas de carro. <br>

O projeto tem como refêrencia a um curso da Alura de Visão Computacional com OpenCv, [clique aqui](https://www.alura.com.br/formacao-visao-computacional-opencv) para ver o curso.

### Instalando as Bibliotecas

``` python
!pip install opencv-python==4.6.0.66
!pip install mediapipe==0.8.11
!pip install numpy==1.22.3
```

## Captura ao vivo com OpenCV <br>

Para ajudarmos os motoristas da melhor forma possível, precisamos que o algoritmo detecte a sonolência em tempo real, para isso, o nosso algoritmo deve realizar a coleta e captura do rosto do motorista em tempo real também. Para isso, utilizamos o pacote do OpenCV. <br>

- Link da documentação do OpenCV: [clique aqui](https://docs.opencv.org/4.x/)

### Importando a Biblioteca

``` python
import cv2
```

### Abrindo a Câmera

``` python
cap = cv2.VideoCapture(0)

# condição que faz a captura ao vivo
while cap.isOpened():

    sucesso, frame = cap.read()

    if not sucesso:

        print('Captura não foi feita')
        continue

    cv2.imshow('Câmera', frame)

    if cv2.waitKey(10) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()
```

O método `VideoCapture` cria um objeto de captura de vídeo utilizando a câmera padrão do computador (índice 0). Caso haja mais de uma câmera, pode-se usar um índice diferente além disso, vamos utilizar mais duas variáveis essenciais:

- __sucesso__: Ela verifica se a variável `cap` tem alguma coleta. Além disso, a variável `sucesso` é do tipo booleano.
- __frame__: Contém o frame capturado. <br>

Seguindo a estrutura de repetição `while` utilizamos mais alguns métodos e funções importantes para a captura: <br>

- __read( )__: Lê um frame da câmera
- __imshow__: Exibe o frame capturado pela câmera em uma janela chamada "câmera"
- __release( )__: Libera o objeto câmera de vídeo, ou seja, ele para de exibir
- __destroyAllWindows( )__: Fecha todas as janelas abertas pelo OpenCV

## Identificando seu rosto com o MediaPipe Face Mesh <br>

Para realizarmos essa identificação, usaremos um framework de código aberto criado pela Google, o MediaPipe. Ele utiliza Machine Learning / Deep Learning para fazer diversas aplicações/soluções com visão computacional. A solução que utilizaremos no nosso projeto será o MediaPipe Face Mesh que se trata de um modelo de visão computacional que detectar o rosto e suas coordenadas.

Link da documentação MediaPipe Face Mesh: [clique aqui](https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md)

### Importando as Bibliotecas

``` python
import mediapipe as mp
```

### Importando as Soluções que vamos utilizar <br>

Existem duas soluções que são mais interessantes para a nossa aplicação:

- __drawing_utils__: Fornece funções úteis para desenhar os resultados da detecção de rosto usando o MediaPipe Face Mesh.
- __face_mesh__: É o modelo de visão computacional do MediaPipe que detecta e rastreia o rosto humano.

``` python
mp_drawing = mp.solutions.drawing_utils

mp_face_mash = mp.solutions.face_mesh
```

### Aplicando o MediaPipe integrado ao OpenCv

Utilizando o código para abrir a câmera que fizemos anteriormente:

``` python
cap = cv2.VideoCapture(0)

    while cap.isOpened():

        sucesso, frame = cap.read()

        if not sucesso:

            print('Captura não foi feita')
            continue

        cv2.imshow('Câmera', frame)

        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
```

Após a variável `cap`, apertaremos __“Enter”__. Em seguida, selecionaremos todo o bloco `while`, isto é, da linha em que criamos o `while` até `break`. Vamos apertar a tecla __“TAB”__ e gerar mais uma identação. Nós aplicaremos o comando `with` para utilizarmos a solução do Face Mesh:

``` python
cap = cv2.VideoCapture(0)

with mp_face_mash.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:

    while cap.isOpened():

        sucesso, frame = cap.read()

        if not sucesso:

            print('Captura não foi feita')
            continue

        cv2.imshow('Câmera', frame)

        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
```

Após o `with`, passamos o objeto que já criamos,` mp_face_mesh`, e chamamos o método `FaceMesh()`. Esse método trará o modelo que vai ler, analisar e identificar a face. Dentro dos parênteses, especificaremos dois parâmetros:

- min_detection_confidence=0.5: o valor mínimo de confiança para detecção de face. Nós atribuímos o valor de 0,5, ou seja, mínimo de 50% de confiança para detectar a face.

- min_tracking_confidence=0.5: o valor mínimo de confiança para detectar os pontos da face, que atribuímos o valor 0,5.

Os parâmetros de detecção de rosto e de rastreamento possuem valores mínimos para que suas detecções/rastreamentos sejam considerados bem-sucedidos. Por padrão, seus valores correspondem a 0.5, porém podem variar de 0.0 a 1.0. Fora dos parênteses, passamos a variável que vai manipular a solução do Face Mesh: `as facemesh`. Terminada essa parte, vamos criar um espaço abaixo do bloco `if not sucesso` e iniciaremos a manipulação do frame.

### BGR para RGB

A primeira manipulação é transformar a cor do frame de BGR para RGB. Sabemos que, no OpenCV, a captura de tela vem em BGR. Mas, para processarmos esse frame no MediaPipe, ele precisa estar em RGB. Por isso, faremos `frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`.

``` python
cap = cv2.VideoCapture(0)

with mp_face_mash.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:

    while cap.isOpened():

        sucesso, frame = cap.read()

        if not sucesso:

            print('Captura não foi feita')
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow('Câmera', frame)

        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
```

Além disso, vamos atribuir uma variável que será a nossa saída: `saida_facemesh`. Essa variável receberá os dados processados do frame. Então, `saida_facemesh` será igual a `facemesh.process(frame)`, sendo frame a variável que será processada.

``` python
cap = cv2.VideoCapture(0)

with mp_face_mash.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:

    while cap.isOpened():

        sucesso, frame = cap.read()

        if not sucesso:

            print('Captura não foi feita')
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        saida_facemesh = facemesh.process(frame)

        cv2.imshow('Câmera', frame)

        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
```

Agora, vamos transformar novamente o frame para BGR, já que o __OpenCV__ trabalha em __BGR__. Basta copiar a linha de frame em que já transformamos a cor e colá-la, alterando o parâmetro de troca.

``` python
cap = cv2.VideoCapture(0)

with mp_face_mash.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:

    while cap.isOpened():

        sucesso, frame = cap.read()

        if not sucesso:

            print('Captura não foi feita')
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        saida_facemesh = facemesh.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Câmera', frame)

        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
```

### Mostrando os pontos detectados

Fizemos a detecção do rosto com `facemesh.process(frame)`. Agora, podemos mostrar essa detecção que o MediaPipe fez. Para isso, utilizaremos um laço de repetição: o `for`. Dentro dele, criaremos a variável `face_landmarks`, referente às coordenadas da nossa face. A variável `face_landmarks` será atribuída ao conjunto de coordenadas que coletaremos com `saida_facemesh` (resultado do nosso processamento). Em seguida, passaremos o método `multi_face_landmarks`, que nos retornará as coordenadas x, y e z de cada ponto que o MediaPipe encontrar no nosso rosto.

``` python
cap = cv2.VideoCapture(0)

with mp_face_mash.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:

    while cap.isOpened():

        sucesso, frame = cap.read()

        if not sucesso:

            print('Captura não foi feita')
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        saida_facemesh = facemesh.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            for face_landmarks in saida_facemesh.multi_face_landmarks:

        cv2.imshow('Câmera', frame)

        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
```

A partir dessa coleta das marcas faciais, queremos que ele desenhe todos os pontos no rosto. Para isso, utilizaremos o método `mp_drawing`.

``` python
cap = cv2.VideoCapture(0)

with mp_face_mash.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:

    while cap.isOpened():

        sucesso, frame = cap.read()

        if not sucesso:

            print('Captura não foi feita')
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        saida_facemesh = facemesh.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            for face_landmarks in saida_facemesh.multi_face_landmarks:

                 mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mash.FACEMESH_CONTOURS)

        cv2.imshow('Câmera', frame)

        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
```

Chamamos o objeto `mp_drawing` e utilizamos o método `draw_landmarks()` para o desenho de cada ponto/coordenada que for coletada do nosso rosto. Dentro dos parênteses, colocaremos o frame, que é o que está sendo coletado, e o `face_landmarks`, que são as coordenadas de cada ponto. Ainda nos parênteses, utilizaremos o` mp_face_mesh.FACEMESH_CONTOURS` para especificar os nossos pontos. <br>
Diante de todas essas modificações, esse código vai nos permitir detectar a face e coletar pontos de olhos, sobrancelhas, bochechas, nariz, boca e outras características do rosto humano.

### Ajustando o código

Dentro do Face Mesh, se a face, ou seja, o rosto da pessoa, não for detectável por uma câmera, podemos ter problemas ou erros na hora da execução. Para resolver isso, basta usar um bloco de tratamento para erros básicos do Python: `try` e `except`. <br> Para realizar essa criação, basta selecionar as duas linhas do `for`, apertar __"TAB"__ para gerar uma identação e, acima do for, colocaremos o primeiro bloco: `try:`. Usamos os "dois pontos (:)" para que ele crie uma estrutura de análise para o laço `for`.

``` python
cap = cv2.VideoCapture(0)

with mp_face_mash.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:

    while cap.isOpened():

        sucesso, frame = cap.read()

        if not sucesso:

            print('Captura não foi feita')
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        saida_facemesh = facemesh.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:

            for face_landmarks in saida_facemesh.multi_face_landmarks:

                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mash.FACEMESH_CONTOURS)

        cv2.imshow('Câmera', frame)

        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
```

Alinhado ao `try`, passaremos o `except:`, que determina como possíveis erros devem ser tratados. No nosso caso, vamos pedir para ele apenas continuar a leitura.

``` python
cap = cv2.VideoCapture(0)

with mp_face_mash.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:

    while cap.isOpened():

        sucesso, frame = cap.read()

        if not sucesso:

            print('Captura não foi feita')
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        saida_facemesh = facemesh.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:

            for face_landmarks in saida_facemesh.multi_face_landmarks:

                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mash.FACEMESH_CONTOURS)
       
        except:
            pass

        cv2.imshow('Câmera', frame)

        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
```

## Ajustando a visualização <br>

Nessa parte vamos melhorar essa visualização para alcançarmos uma análise mais assertiva sobre o que está acontecendo com os pontos na nossa face. Então, vamos alterar os pontos e como são mostrados. <br>

### Alterando cor, grossura e tamanho dos círculos

Existe uma opção no MediaPipe que nos permite especificar a cor, a grossura e o tamanho do círculo de cada ponto do nosso rosto, bem como a linha branca que está em volta da nossa face. Para modificar essa parte precisamos retomar na linha aonde criamos o desenho das nossas coordenadas:

```python
       try:

            for face_landmarks in saida_facemesh.multi_face_landmarks:

                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mash.FACEMESH_CONTOURS)
       
        except:
            pass
```

Nossa primeira alteração será nos pontos em vermelho. Podemos deixá-los menores e preenchidos com outra cor. Para isso, usaremos o parâmetro `landmark_drawing_spec`, que será adicionado ao parâmetro `drawn_landmarks()`. Ao final dos parâmetros que já estão no método, passaremos vírgula e apertaremos __"Enter"__. Também apertaremos __"TAB"__ para que o comando não fique na mesma identação. Finalmente, adicionaremos o `landmark_drawing_spec`:

```python
       try:

            for face_landmarks in saida_facemesh.multi_face_landmarks:

                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mash.FACEMESH_CONTOURS
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255, 102, 102), thickness = 1, circle_radius = 1))
       
        except:
            pass
```

Esse é o primeiro parâmetro com o qual trabalharemos para alterar a visualização dos pontos no nosso rosto. Após isso, vamos chamar novamente o objeto `mp_drawing` e, em seguida, usaremos o método `DrawingSpec()`. Dentro do `DrawingSpec`, vamos adicionar mais alguns parâmetros: <br>

- __color__: o código RGB da cor das linhas que serão desenhadas nos pontos.
- __thickness__: a espessura das linhas.
- __circle_radius__: o raio dos círculos que representam os pontos.

### Alterando a grossura das linhas e a cor:

A grossura das linhas de conexão ainda incomoda, e a cor branca também. Podemos alterar os parâmetros de conexão. Para isso, usaremos outro parâmetro, bastante similar ao `landmark_drawing_spec`, o `connection_drawing_spec`. Após o sinal de __"igual (=)"__, vamos atribuir exatamente o que atribuímos no parâmetro de `landmark_drawing_spec`. Então, vamos apenas copiar o código que já construímos:

```python
try:

    for face_landmarks in saida_facemesh.multi_face_landmarks:

        mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mash.FACEMESH_CONTOURS,
                                landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255, 102, 102), thickness = 1, circle_radius = 1),
                                connection_drawing_spec = mp_drawing.DrawingSpec(color=(102, 204, 0), thickness = 1, circle_radius = 1))
except:
    pass
```

Só vamos alterar os valores do parâmetro das cores, porque não queremos que as linhas sejam azuis, mas, sim, uma tonalidade mais esverdeada, por isso, precisamos diminuir o valor da cor azul para 102 e deixar o valor de verde mais forte, com 204. Por fim, vamos zerar a cor vermelha.

## Coordenadas da face

Fazendo uma analogia simples para entender melhor como que o MediaPipe identifica o nosso rosto, imagine que o seu rosto é como um mapa com vários pontos de referência. Esses pontos de referência são os diferentes elementos do seu rosto, como os olhos, nariz, boca, sobrancelhas, etc. Cada um desses elementos pode ser representado por coordenadas, assim como em um mapa. O MediaPipe, que é a ferramenta utilizada nesta aula, é como um GPS que consegue identificar e mapear esses pontos de referência do seu rosto. Quando você coloca o seu rosto na frente da câmera, o MediaPipe "enxerga" o seu rosto e consegue detectar com precisão a localização de cada um desses pontos.

```python

[landmark {
   x: 0.57531583
   y: 0.527997806
   z: -0.037834693
 }
 landmark {
   x: 0.575518
   y: 0.4628154
   z: -0.077181056
 }
 landmark {
   x: 0.5743907
   y: 0.48317027
   z: -0.039508026
 }
 landmark {
   x: 0.5586666
   y: 0.3892232
   z: -0.06116421
 }
 landmark {
   x: 0.5750047
   y: 0.43989837
   z: -0.082985066
 }
...
landmark {
   x: 0.6852603
   y: 0.3033984
   z: 0.019007659
 }]

```

Essas coordenadas detectadas pelo MediaPipe são como um conjunto de números que indicam a posição exata de cada ponto do seu rosto. Imagine que cada ponto do seu rosto tem um __"endereço"__ composto por três números: um para a posição horizontal (eixo x), outro para a posição vertical (eixo y) e outro para a profundidade (eixo z). Então, quando se pegarmos e execurtamos separadamente o código `saida_facemesh.multi_face_landmarks`, vamos está acessando essa __"lista de endereços"__ que o MediaPipe gerou para mapear todos os pontos do seu rosto. Essa lista contém as coordenadas de cada ponto, como se fosse um catálogo com os __"endereços"__ de todos os elementos do seu rosto.

## Análise dos olhos

Avançando um pouco mais, o próximo passo agora é fazer uma análise dos nossos olhos. Eles são bons indicadores de sono. Quando estamos com sono, nossos olhos ficam fechados por mais tempo e a piscada demora um pouco mais. O artigo __Real-Time Eye Blink Detection using Facial Landmarks__ (em tradução livre: Detecção de piscar de olhos em tempo real usando coordenadas faciais), escrito por __Tereza Soukupova__ e __Jan Cechnos__, nos fornece duas informações principais:

1. Existem 6 pontos em cada olho: um em cada extremidade dos olhos, dois na parte de cima e dois na parte de baixo. Considerando os dois olhos, são 12 pontos no total.

2. É o valor de __EAR__, uma equação que utilizaremos para verificar se os olhos estão abertos ou fechados. __EAR__ significa __Eye Aspect Ratio__ e se refere à distância dos olhos. <br>

O __EAR__ é dado pelo valor da distância euclidiana entre o ponto P2 e o ponto P6, mais o valor da distância euclidiana entre o ponto P3 e o ponto P5. Isso tudo sobre duas vezes a distância euclidiana entre o ponto P1 e o ponto P4.

Antes de iniciar outra parte vou deixar o link desse artigo [clique aqui](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf) para acessar e entender melhor. <br>

### Criação dos pontos

Tem outra imagem que representa mais ou menos os 12 pontos, mas com valores, acredito que dessa forma dá para entender melhor. Além disso, vamos utilizar esses valores para seguir com o nosso código.

### Criando uma lista com os valores e depois concatenando

A partir desso ponto nós criaremos uma lista que contenha cada um desses 6 pontos. Para isso, no inicio do código vamos criar duas variáveis __p_olho_esq__  e __p_olho_dir__ que representa os pontos dos olhos esquerdos e direito.

```python
import cv2
import mediapipe as mp
import numpy

mp_drawing = mp.solutions.drawing_utils

mp_face_mash = mp.solutions.face_mesh

p_olho_esq = [385, 380, 387, 373, 362, 263]
p_olho_dir = [160, 144, 158, 153, 33, 133]

```

Agora vamos concatenar essa duas listas em uma variável chamada `pontos_olhos`:

```python
p_olho_esq = [385, 380, 387, 373, 362, 263]
p_olho_dir = [160, 144, 158, 153, 33, 133]

pontos_olhos = p_olho_esq + p_olho_dir
```

### Criando a visualização no OpenCV

Criamos as listas com todos os pontos dos nossos olhos. Falta criar uma visualização no OpenCV para, de fato, verificarmos essa transformação. Vamos adicionar um novo código abaixo da verificação de `sucesso`. Após o continue, vamos apertar __“Enter”__ e, fora do bloco de condicional, nós coletaremos o tamanho, isto é, a `largura` e `comprimento` do vídeo. O motivo de fazer isso é porque precisamos de um parâmetro para transformar os pontos normalizados do MediaPipe em pontos de pixel.

```python
if not sucesso:

            print('Captura não foi feita')
            continue

        comprimento, largura, _ = frame.shape
```

Isso nos permitirá colocar os pontos dos olhos esquerdo e direito na imagem. Além disso, com os valores de comprimento e largura, já podemos identificar quais são as coordenadas que correspondem aos pontos dos nossos olhos e executar a desnormalização desses pontos. Até porque, o MediaPipe retorna as coordenadas x, y e z em pontos normalizados, mas para que o MediaPipe reproduza algum ponto ou círculo, isto é, marque pontos no rosto, ele precisa que esses pontos estejam em pixels, não normalizados. <br>

Dentro do bloco de `try`, abaixo da parte em que desenhamos nossas coordenadas com MediaPipe, vamos apertar __"Enter"__, voltar para a identação, e faremos algo similar à aula anterior: coletar o `face_landmarks` e enumerar os pontos com o `for`. Vamos também inserir o código que pega as coordenadas que vimos antes, no final, o bloco try vai ficar dessa forma:

```python
try:

            for face_landmarks in saida_facemesh.multi_face_landmarks:

                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mash.FACEMESH_CONTOURS,
                                          landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255, 102, 102), thickness = 1, circle_radius = 1),
                                          connection_drawing_spec = mp_drawing.DrawingSpec(color=(102, 204, 0), thickness = 1, circle_radius = 1))
               
                face = face_landmarks.landmark

                for id_coord, coord_xyz in enumerate(face):

                    if id_coord in p:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, frame.shape[1], frame.shape[0])

                        if coord_cv:
                            cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)
               
        except:
            pass
```

### Explicando o que fizemos

Vamos analisar esse trecho de código passo a passo: <br>

1. __face = face_landmarks.landmark__: Essa linha armazena as coordenadas normalizadas (entre 0 e 1) de todos os pontos do rosto detectados em uma variável chamada face.

2. __for id_coord, coord_xyz in enumerate(face):__: Esse loop percorre cada um dos pontos do rosto armazenados em face, obtendo tanto o índice do ponto `(id_coord)` quanto suas coordenadas normalizadas `(coord_xyz)`.

3. __if id_coord in p:__: Essa condição verifica se o índice do ponto atual `(id_coord)` está presente na lista p_olhos que você definiu anteriormente. Isso é feito para selecionar apenas os pontos dos olhos.

4. __coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, frame.shape[1], frame.shape[0])__: Essa linha converte as coordenadas normalizadas dos pontos dos olhos `(coord_xyz.x e coord_xyz.y)` para coordenadas de pixel, usando a largura e altura do frame `(frame.shape[1] e frame.shape[0])`. O resultado é armazenado na variável `coord_cv`.

5. __if coord_cv:__: Essa condição verifica se a conversão de coordenadas foi bem-sucedida (ou seja, se as coordenadas de pixel estão dentro dos limites da imagem).

6. __cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1):__ Essa linha desenha um círculo azul (RGB: 255, 0, 0) com raio 2 pixels no ponto do olho, usando as coordenadas de pixel obtidas anteriormente.

## Cálculo do EAR

Encontramos as coordenadas de cada um dos olhos para calcularmos o valor de EAR e agora vamos, de fato, programar o cálculo desse índice.<br>

### Importando as Bibliotecas

O primeiro passo para a realização do cálculo é importar a biblioteca NumPy.


```python
import numpy as np
```

### Criando a função

Vamos criar uma função que fará todo esse cálculo do EAR. Para isso, vamos criar um novo arquivo .py chamado __funcao_calculo_ear__, onde deixaremos nossa função, [clique aqui](funcao_calculo_ear.py) para acessar o arquivo da função. Então basicamente essa parte da função ficou dessa forma: <br>

1. __face = np.array([[coord.x, coord.y] for coord in face]):__ Essa linha converte as coordenadas da face, que estão em formato de objeto, em um array NumPy. Isso facilita os cálculos posteriores.

2. __face_esq = face[p_olho_esq,:]__ e __face_dir = face[p_olho_dir,:]:__ Essas linhas selecionam as coordenadas dos pontos dos olhos esquerdo e direito, respectivamente, a partir do array face.

3. Essas linhas calculam o EAR (Eye Aspect Ratio) para o olho esquerdo e o olho direito, respectivamente:

    ```python
    ear_esq = (np.linalg.norm(face_esq[0]-face_esq[1])+np.linalg.norm(face_esq[2]-face_esq[3]))/(2*(np.linalg.norm(face_esq[4]-face_esq[5])))
   
    ear_dir = (np.linalg.norm(face_dir[0]-face_dir[1])+np.linalg.norm(face_dir[2]-face_dir[3]))/(2*(np.linalg.norm(face_dir[4]-face_dir[5])))
    ```

    O EAR é calculado usando a distância euclidiana entre os pontos dos olhos, conforme explicado na aula. Essa distância é normalizada pela distância entre os últimos pontos do olho, multiplicada por 2.

4. __except: ear_esq = 0.0; ear_dir = 0.0__: Caso ocorra algum erro durante o cálculo do EAR, essa parte do código atribui o valor 0.0 aos valores do EAR esquerdo e direito.

5. __media_ear = (ear_esq+ear_dir)/2__: Essa linha calcula a média entre o EAR do olho esquerdo e do olho direito.

6. __return media_ear__: Finalmente, a função retorna a média do EAR.

### Inserindo a função no código principal

Agora dentro da estrutura do try, vamos inserir um novo código após o `cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)`. Essa nova linha de código fara o cálculo do EAR.

 ```python
       try:

            if coord_cv:
                cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)

            ear = calculo_ear(face,p_olho_dir, p_olho_esq)
               
        except:
            pass
```

Ok, agora para mostrar esse resultado vamos criar um retângulo que será o fundo do nosso texto. Nós usaremos `cv2.retangle()` para isso. Nos parênteses, definiremos o lugar em que ele ficará: no nosso `frame`. Também indicaremos o ponto inicial onde esse retângulo será criado e seu ponto final: `(0,1),(290,140)`. Fora dos parênteses, especificaremos a cor desse parâmetro, que será um tom de cinza, portanto: `(58,58,55)`.

 ```python
       try:

            if coord_cv:
                cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)

            ear = calculo_ear(face,p_olho_dir, p_olho_esq)

            cv2.rectangle(frame, (0 , 1), (290 , 140), (58 , 58 , 55 ), -1)
               
        except:
            pass
```

Agora vamos adicionar um texto referente ao valor do EAR.

 ```python
       try:

            if coord_cv:
                cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)

            ear = calculo_ear(face,p_olho_dir, p_olho_esq)

            cv2.rectangle(frame, (0 , 1), (290 , 140), (58 , 58 , 55 ), -1)

            cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.9, (255, 255, 255), 2)
               
        except:
            pass
```

O comando para adicionar o texto, é `cv2.putText()`, que é o método do OpenCV para inserir texto. Nos parênteses, passamos o objeto frame, e o texto, composto de EAR: e o valor resultante do EAR. Usamos a função round para que um valor muito grande não seja exposto na tela. Entre parênteses, passamos ear, 2, porque apenas duas casas decimais devem ser exibidas. Em seguida, especificamos o ponto inicial em que o texto vai aparecer: `(1,24)`. Também especificamos a fonte, `cv2.FONT_HERSHEY_DUPLEX`, o tamanho do texto, `0.9`, e a cor, que será branca, `(255, 255, 255)`. Por fim, especificamos a grossura da letra, `2`.

## Cálculo do Tempo

Se estamos sonolentos, com vontade de dormir, tendemos a fechar os olhos por mais tempo ou a fechá-los de vez e dormir. Quando construímos um código de análise da sonolência através dos olhos, precisamos verificar quanto tempo a pessoa que está sendo analisada passa de olhos fechados. Nós podemos fazer isso, pois já sabemos calcular e verificar o valor de EAR. <br> Vamos trabalhar com o tempo em que os olhos ficam fechados e o tempo em que o valor de EAR estará menor. Isso porque quando o EAR está alto, significa que o olho está aberto, mas quando ele diminui, quer dizer que o olho está fechado. <br>

Agora que sabemos disso, vamos começar a trabalhar com o tempo. Vamos ter que importar a __biblioteca time__, que lida com o tempo.

### Importando a Biblioteca

 ```python
import time
```

### Criando algumas variáveis:

Antes do `cap`, que é quando inicia o código do OpenCV, nós vamos criar a variável ear_limiar e atribuir a ela o valor de __0.3__, que basicamente indica que estamos com os olhos abertos. Esse valor foi retirado do cálculo do EAR que fizemos anteriormente. Além disso, vamos criar a variável `dormindo` que vai nos indicar quando que o motorista estiver de olhos fechados, vamos iniciar ela com o valor _0_.  O zero vai representar __"false"__, isto é, quando a pessoa não está dormindo, portanto, não está com o olho fechado. Ela nos ajudará a controlar em que momentos calcularemos o tempo.

 ```python
ear_limiar = 0.3

dormindo = 0

cap = cv2.VideoCapture(0)
```

### Inserindo no nosso código principal

Após a criação das duas variáveis o próximo passo agora seria localizar o `putText()` no código. Após o `putText()`, vamos inserir uma condicional que verificará se o valor de EAR está abaixo ou acima do limiar que especificamos.

```python
cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9, (255, 255, 255), 2)

if ear < ear_limiar

```

Estamos realizando a seguinte verificação: se o valor de `EAR` capturado em tempo real é menor que o `ear_limiar`, então, o olho está fechado. Pode ser uma piscada ou porque a pessoa fechou o olho para dormir. Quando isso acontecer, é interessante alterar a flag dormindo para 1, valor positivo, ou seja, a pessoa fechou os olhos.

```python
cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9, (255, 255, 255), 2)

if ear < ear_limiar:

    dormindo = 1

```

Além de verificar se `dormindo` é igual a 1, vamos inicializar o tempo, ou seja, o tempo em que o olho vai ficar fechado. Isso pode nos ajudar a detectar sonolência, até porque, quando estamos sonolentos, tendemos a fechar os olhos por mais tempo. Sendo assim, antes de `dormindo = 1`, vamos criar uma variável que vai salvar o tempo inicial. Nós a nomearemos como `t_inicial`. Ela será igual a `time.time()`. Com isso, coletaremos o tempo real, marcado a partir dessa biblioteca. <br>
Esse tempo não pode ser coletado todas as vezes em que estivermos com os olhos fechados. Precisamos de um limite e que esse cálculo ocorra apenas quando o olho sair do estado aberto para fechado. Para incluirmos esse limite, usaremos um operador ternário.

```python
cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9, (255, 255, 255), 2)

if ear < ear_limiar:

    t_inicial = time.time() if dormindo == 0
    dormindo = 1

```

Estamos dizendo que ele deve contar `time.time()` apenas quando o olho estiver aberto. Isso indica que sairá de dormindo igual a zero, quando o olho está aberto, para dormindo igual a um, quando o olho está fechado. Por isso, podemos contar o tempo. Caso isso não aconteça, diremos que `t_inicial` é igual ao `t_inicial`.

```python
cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9, (255, 255, 255), 2)

if ear < ear_limiar:

    t_inicial = time.time() if dormindo == 0 else t_inicial
    dormindo = 1

```

Depois disso, vamos criar uma condicional/limite para quando a pessoa abrir os olhos. Neste momento, precisamos parar a contagem. Adicionaremos outra condicional, desta vez, para "olho fechado".

```python
cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9, (255, 255, 255), 2)

if ear < ear_limiar:

    t_inicial = time.time() if dormindo == 0 else t_inicial
    dormindo = 1

if dormindo == 1 and ear >= ear_limiar:

```

Com esse trecho de código, estamos dizendo que: se o olho estiver fechado e o `EAR` superior ao limiar, ou seja, olho aberto, quer dizer que o olho já não está fechado, portanto, a pessoa acordou ou piscou. Portanto, temos que setar a variável dormindo para zero novamente.

```python
cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9, (255, 255, 255), 2)

if ear < ear_limiar:

    t_inicial = time.time() if dormindo == 0 else t_inicial
    dormindo = 1

if dormindo == 1 and ear >= ear_limiar:

    dormindo = 0
                   
    t_final = time.time

```

### Calculando o tempo que o olho ficou fechado

Agora podemos calcular o tempo geral, isto é, o tempo em que o olho ficou fechado. Para isso, criaremos uma variável chamada tempo. Esse tempo será igual ao tempo final (o tempo em "tempo real") menos o tempo inicial (o tempo marcado toda vez que fechamos os olhos).

```python
tempo = (t_final-t_inicial)
```

Isso não acontecerá sempre, até porque, o olho não fechará a todo momento. A pessoa que está dirigindo pode estar olhando para a rua, normalmente, com o olho aberto. Precisamos, mais uma vez, especificar um limite. Vamos inserir um operador ternário novamente: `if`. E a condicional será dormindo (variável de verificação de olho) __"igual igual"__ a 1. Quer dizer que, se olho estiver fechado, ele vai ficar calculando o tempo em que o olho está fechado. Se o dormindo não for __"igual igual"__ a 1, o tempo vai ser igual a zero. Ou seja, a pessoa está com o olho aberto, logo, não contaremos o tempo de olho fechado.

```python
tempo = (t_final-t_inicial) if dormindo == 1 else 0.0
```

### Mostrando o valor do tempo

Vamos imprimir esse valor de tempo com o putText(), usaremos a mesma estrutura do cv2.putText() que fizemos para o __EAR__. Ainda na estrutura do `try`, abaixo da linha de código que calcula o tempo vamos inserir essas outras linhas: <br>

```python
tempo = (t_final-t_inicial) if dormindo == 1 else 0.0

cv2.putText(frame, f"Tempo: {round(tempo, 3)}", (1, 80),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9, (255, 255, 255), 2)
```

Tá, mas o que esse código está fazendo? Vou deixar uma explicação aqui.

1. __cv2.putText(frame, f"Tempo: {round(tempo, 3)}", (1, 80)__:

    - __cv2.putText()__ é uma função da biblioteca OpenCV que permite escrever texto em uma imagem.
    - __frame__ é a imagem capturada pela câmera, onde o texto será exibido.
    - __f"Tempo: {round(tempo, 3)}"__ é a string que será exibida. Ela contém a palavra "Tempo:" seguida do valor da variável tempo, arredondado com 3 casas decimais.
    - __(1, 80)__ são as coordenadas `(x, y)` na imagem onde o texto será posicionado, neste caso, na posição `(1, 80)`.

2. __cv2.FONT_HERSHEY_DUPLEX__: Este parâmetro especifica o tipo de fonte a ser utilizada para o texto.

3. __0.9__: Este parâmetro define o tamanho da fonte do texto.

4. __(255, 255, 255)__: Este parâmetro define a cor do texto, neste caso, branco `(RGB: 255, 255, 255)`.

5. __2)__: Este parâmetro define a espessura da fonte do texto.

Com isso tudo, já podemos adicionar um alerta, caso o tempo com os olhos fechados demore, aumentando o risco de acidentes. É bem simples adicionar esse alerta, vamos usar uma condicional, caso o tempo extrapole um limite específico: 2 segundos.

```python
if tempo >= 2 :
```

Caso esse tempo extrapole vamos mostrar uma mensagem de texto na tela.

```python
 if tempo>=1.5:
   
    cv2.rectangle(frame, (30, 400), (610, 452), (109, 233, 219), -1)
    cv2.putText(frame, f"Muito tempo com olhos fechados!", (80, 435),
                cv2.FONT_HERSHEY_DUPLEX,
                0.85, (58,58,55), 1)

```

Estamos reutilizando a estrutura que utilizamos para mostrar o valor do tempo, mas com algumas alterações. Vou deixar uma explicação do código:

1. __cv2.rectangle(frame, (30, 400), (610, 452), (109, 233, 219), -1)__:
   
    - Esta linha desenha um retângulo na imagem `frame`.
    - As coordenadas `(30, 400)` e `(610, 452)` definem os pontos superior esquerdo e inferior direito do retângulo, respectivamente.
    - A cor do retângulo é definida pelo parâmetro `(109, 233, 219)`, que representa a cor azul claro no formato RGB.
    - O parâmetro `-1` preenche o retângulo com a cor especificada.

2. __cv2.putText(frame, f"Muito tempo com olhos fechados!", (80, 435)__:

    - Esta linha adiciona um texto na imagem `frame`.
    - A string __"Muito tempo com olhos fechados!"__ é o texto que será exibido.
    - As coordenadas `(80, 435)` definem a posição do texto na imagem, com o canto superior esquerdo do texto nessa posição.


3. __cv2.FONT_HERSHEY_DUPLEX__: Este parâmetro especifica o tipo de fonte a ser utilizada para o texto.

4. __0.85__: Este parâmetro define o tamanho da fonte do texto.

5. __(58,58,55)__: Este parâmetro define a cor do texto, neste caso, um cinza escuro no formato RGB.

6. __1)__: Este parâmetro define a espessura da fonte do texto.

## Identificando e manipulando a boca

Até agora na construção do nosso projeto, conseguimos:

- Identificar os pontos dos olhos.
- Definir um índice para verificar se olho está aberto.
- Produzir uma mensagem de alerta se olho ficar fechado muito tempo.

Algo recomendado para qualquer projeto é realizar uma análise: verificar possíveis problemas e pensar em ajustes para eles. Se rodarmos o código feito até agora, vamos perceber que, se estivermos sorrindo e dando uma gargalhada, nossos olhos ficarão semicerrados. Isso faz com que nosso código entenda que os olhos estão fechados. <br> Para corrigir isso, vamos precisar verificar os pontos da boca, que no total são 8 pontos.

Nós realizaremos um cálculo similar ao __EAR__, utilizando esses pontos. O__ MAR__ ou __Mouth Aspect Ratio__ (Em tradução livre: Proporção da boca) funciona exatamente como o EAR, mas para as coordenadas da boca. <br>

### Mostrando e verificando os pontos da boca

O primeiro passo agora é montar uma lista com os valores da boca, assim como fizemos para os olhos

```python
p_boca = [82, 87, 13, 14, 312, 317, 78, 308]

cap = cv2.VideoCapture(0)
```

Após isso, adicionaremos esses pontos na nossa visualização. Nós já preparamos um código de verificação das coordenadas presentes nos pontos dos olhos. A partir desses valores, desnormalizamos o valor das coordenadas e colocamos um círculo em cada uma delas.

```python
  if id_coord in p_olhos:
        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x,coord_xyz.y, largura, comprimento)
        cv2.circle(frame, coord_cv, 2, (255,0,0), -1)

```

Vamos trocar o `p_olhos` para `p_boca`

```python
  if id_coord in p_boca:
        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x,coord_xyz.y, largura, comprimento)
        cv2.circle(frame, coord_cv, 2, (255,0,0), -1)
```

Assim, se executarmos o código ele já vai estar identificando a boca.

## Cálculo do MAR

Encontramos os pontos na nossa face e agora vamos definir o valor de __MAR (Mouth Aspect Ratio)__ dentro do nosso código para aprimorá-lo cada vez mais.

A equação é muito similar ao EAR. O cálculo dos pontos ligados verticalmente se dá pela distância euclidiana entre eles. São seis pontos: 82 e 87; 13 e 14; e 312 e 317. Somamos os valores das distâncias euclidianas e dividimos por duas vezes a distância euclidiana entre os pontos das extremidades: 78 e 308.

### Criação da função MAR

Faremos o cálculo de maneira bastante similar. Nós criaremos uma linha de código para a nossa função `def calculo_mar()`. O `calculo_mar()` receberá os pontos da face e da boca. Assim como fizemos na função anterior, vamos criar um arquivo separdo para a função e depois importa-lá aqui no código principal.

- Link para a função: [clique aqui](funcao_calculo_mar.py)

Vou deixar o código da função aqui e logo em seguida a explicação dessa função.

```python
def calculo_mar(face,p_boca):

    try:

         face = np.array([[coord.x, coord.y] for coord in face])
         
         face_boca = face[p_boca, :]

         mar = (np.linalg.norm(face_boca[0] - face_boca[1])
                + np.linalg.norm(face_boca[2] - face_boca[3])
                + np.linalg.norm(face_boca[4] - face_boca[5])) / (2*(np.linalg.norm(face_boca[6] - face_boca[7])))
   
    except:
         
         mar = 0.0
   
    return mar
```

Como que funciona essa função? <br>

A função `calculo_mar()` tem o objetivo de calcular o valor do __Mouth Aspect Ratio (MAR)__, que é uma medida que indica a abertura da boca.

1. __def calculo_mar(face, p_boca)__: Esta é a definição da função `calculo_mar()`, que recebe dois parâmetros:

    - __face__: um array Numpy contendo as coordenadas `(x, y)` de todos os `468` pontos de referência da face detectada.
    - __p_boca__: uma lista com os índices dos pontos de referência que correspondem à boca.

2. __try__ : Esse bloco `try` é usado para lidar com possíveis erros que podem ocorrer durante o cálculo do MAR.

3. __face = np.array([[coord.x, coord.y] for coord in face])__ : Essa linha converte o objeto face em um `array Numpy` de duas dimensões, onde cada linha representa as coordenadas `(x, y)` de um ponto de referência.

4. __face_boca = face[p_boca, :]__ : Essa linha cria uma nova variável face_boca, que é um subconjunto do `array face`, contendo apenas os pontos de referência da boca, de acordo com os índices passados no parâmetro `p_boca`.

5. __mar = (np.linalg.norm(face_boca[0] - face_boca[1]) + np.linalg.norm(face_boca[2] - face_boca[3]) + np.linalg.norm(face_boca[4] - face_boca[5])) / (2*(np.linalg.norm(face_boca[6] - face_boca[7])))__ : Essa é a fórmula de cálculo do MAR. Ela calcula:

    - As distâncias euclidianas entre os pares de pontos da boca `(82-87, 13-14 e 312-317)` usando a função `np.linalg.norm()`.
    - A distância euclidiana entre os pontos das extremidades da boca `(78 e 308)` usando a função `np.linalg.norm()`.
    O resultado é a divisão da soma das três primeiras distâncias pela duas vezes a última distância.

6. __except__ : Esse bloco except é usado para lidar com possíveis erros que podem ocorrer durante o cálculo do MAR, como divisão por zero.

7. __mar = 0.0__ : Se ocorrer um erro, o valor do MAR é definido como 0.0.

### Colocando a função no nosso código principal

```python
 ear = calculo_ear(face,p_olho_dir, p_olho_esq)

cv2.rectangle(frame, (0,1),(290,140),(58,58,55),-1)
cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9, (255, 255, 255), 2)
```

Abaixo da coleta de `ear`, faremos a coleta do `mar`. Passaremos `mar` igual ao `calculo_mar()`. Nos parênteses, enviaremos a `face`, que são as nossas coordenadas, e `p_boca`, que é a lista de pontos da boca.

```python
mar = calculo_mar(face,p_boca)

cv2.putText(frame, f"MAR: {round(mar, 2)} {'Aberto' if mar>=mar_limiar else 'Fechado'}", (1, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9, (255, 255, 255), 2)

```

Agora se executarmos o código novamente vamos ver que aparece o cálculo do valor de MAR integrado ao nosso projeto.

### Verificação da boca

Conseguimos definir o valor de `MAR` e mostrá-lo na tela. Falta integrar o resultado desse valor na análise de verificação do olho, isto é, se ele está fechado ou aberto. Para fazer isso vamos criar mais um váriavel no inicio do nosso código atribuir um valor de `0.1` que é valor que indica que a boca está fechada.

```python
mar_limiar = 0.1

cap = cv2.VideoCapture(0)
```

Criada a variável, podemos adicionar limitadores para a verificação de olhos abertos e fechados. Primeiro, incluiremos um alerta da interpretação do valor de `mar`. No `putText()`, ao final das aspas duplas, adicionaremos chaves e, dentro delas, passaremos um operador ternário que retornará um resultado Aberto se o mar (valor resultante do cálculo de MAR) estiver maior ou igual ao `mar_limiar`. Caso não esteja, ele retornará Fechado.

```python
mar = calculo_mar(face,p_boca)

cv2.putText(frame, f"MAR: {round(mar, 2)} {'Aberto' if mar>=mar_limiar else 'Fechado'}", (1, 50),
    cv2.FONT_HERSHEY_DUPLEX,
    0.9, (255, 255, 255), 2)
```

Com isso, conseguimos mostrar se a boca está aberta ou fechada. Em seguida, vamos adicionar à condicional "se os olhos estão abertos ou fechados" outra condicional para que essa verificação seja dada como um olho realmente fechado, piscando ou até com sono, dormindo se o MAR estiver pequeno, isto é, inferior ao limite. E neste caso, a boca estará fechada.

```python
 if ear < ear_limiar and mar < mar_limiar:
   
    t_inicial = time.time() if dormindo == 0 else t_inicial
    dormindo = 1
   
if dormindo == 1 and ear >= ear_limiar:
   
    dormindo = 0

t_final = time.time()

```

Estamos dizendo que se o valor de EAR estiver menor que o limiar, ou seja, se o olho estiver fechado, e o valor de MAR estiver menor que o valor de `mar_limiar`, ou seja, se a boca estiver fechada, significa que a pessoa piscou ou dormiu e o tempo pode começar a rodar. <br>

Por fim, adicionaremos uma condicional que verifica se abrimos o olho. O motivo é que pode acontecer de o código verificar que fechamos os olhos antes de abrirmos a boca. Isso indicará que os olhos `(ear_limiar)` estão fechados e a boca está fechada também `(mar_limiar)`. Então, ele vai começar a contar o tempo, como se estivéssemos dormindo, e não vai parar até que os olhos estejam abertos. Enfim, mesmo se continuarmos sorrindo, ele contará o tempo como se estivéssemos dormindo. <br>

Para que isso não aconteça, nós deixaremos a primeira condição com `and` entre parênteses e adicionaremos outra condição com `or`. O `or` indica que pode acontecer a primeira condição, em que o dormindo é igual a 1 e o ear maior que o limiar (boca aberta) ou a segunda condição, em que o `ear` é menor ou igual ao `ear_limiar` (olho fechado ou semi cerrado) e o mar estará maior ou igual ao `mar_limiar` (olho fechado e boca aberta, por exemplo, durante uma gargalhada). <br>

Neste último caso, a contagem do tempo vai parar e receberemos uma flag `dormindo = 0`.

```python

 if ear < ear_limiar and mar < mar_limiar:
   
    t_inicial = time.time() if dormindo == 0 else t_inicial
    dormindo = 1

if (dormindo == 1 and ear >= ear_limiar) or (ear <= ear_limiar and mar>= mar_limiar):

    dormindo = 0

t_final = time.time()

```

Agora se executarmos o código novamente vamos ver que conseguimos mostrar que estamos de boca aberta ou fechada e se a pessoa sorrir ou abrir a boca, ele não conta o tempo, mas se estivermos de boca fechada e de olhos fechados, ele conta o tempo.

## Explorando pontos de melhoria

Já construímos um código interessante, verificamos os pontos dos nossos olhos para identificar se eles estão abertos ou fechados e há quanto tempo. Além disso, aprimoramos a detecção de olhos fechados ou abertos por meio da integração dos pontos da boca. No entanto, podemos aprimorar o nosso protótipo de detecção de sono ainda mais. Para isso, podemos buscar estudos da comunidade científica. <br>

Tem um artigo chamada _"Drowsiness Detection According to the Number of Blinking Eyes Specified from Eye Aspect Ratio Value Modification"_, que foi escrito por __Novie Pasaribu__, __Agus Prijono__, __Ratnadewi__, __Roy Adhie__ e __Joseph Felix__. Nesse trabalho, a sonolência é verificada a partir do número de piscadas que o olho dá. Essas piscadas são verificadas, por sua vez, a partir do valor de EAR em cada motorista. <br>

O EAR é modificado para levar em conta as diferentes espessuras dos olhos das pessoas. O estudo estabelece que os motoristas são considerados sonolentos se:

- Os olhos ficarem fechados por mais de 45 frames
- Piscarem menos de 10 vezes por minuto

Essas informações sobre os sinais de sonolência são relevantes para o projeto em questão, que busca aprimorar sua detecção de sono adicionando esses parâmetros. Vou deixar o link do artigo para quem quiser dar uma olhada.

- Link artigo: [clique aqui](https://www.atlantis-press.com/proceedings/iclick-18/125913292)

## Contagem de Piscadas

Para podermos aplicar o que conhecemos no estudo de detecção de sonolência, podemos começar adicionando um contador de piscadas ao nosso código. Para fazer isso, vamos criar uma variável contadora chamada contagem_piscadas, deixando-a igual a zero. O trecho em questão ficará da seguinte forma:

```python
contagem_piscadas = 0

cap = cv2.VideoCapture(0)
```

Feito isso, vamos até o bloco condicionaç `try` lá verificaremos quando o olho fecha, que corresponde ao cenário em que a condição do EAR é inferior ao EAR limiar, ou seja, `ear < ear_limiar`, e a pessoa está de boca fechada, com `mar < mar_limiar`. Após a linha do `t_inicial` e antes da linha que contém `dormindo = 1`, inseriremos um espaço e escreveremos um código para incrementar a nossa variável contadora, inserindo `contagem_piscadas = contagem_piscadas+1`. Em teoria, isso incrementa a nossa variável contadora toda vez que piscarmos os olhos.

```python
if ear < ear_limiar and mar < mar_limiar:
    t_inicial = time.time() if dormindo == 0 else t_inicial
    contagem_piscadas = contagem_piscadas+1
    dormindo = 1
```

### Criando um controle

Para termos um controle das piscadas, inseriremos um texto que nos mostra a quantidade de piscadas. Para isso, inseriremos um espaço antes de tempo.

```python
cv2.putText(frame, f"Piscadas: {contagem_piscadas}", (1, 120),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9, (109, 233, 219), 2)
```

Além disso, vamos preciasr adicionar um limitador dessa contagem porque o contador verifica toda vez que eu fecho os olhos por alguns segundos, o número de piscadas é incrementado infinitamente. <br> Para resolver isso, vamos adicionar um operador ternário, que atuatá de forma parecida com aquele em `t_inicial`. Ele executará a ação só quando tivermos uma troca do estágio de olho aberto para o de olho fechado. <br> Assim, colocaremos um  `f` após  `ontagem_piscadas = contagem_piscadas+1`, inserindo a condição `dormindo == 0`. Se dormindo for igual a 1, abarcaremos esse cenário com um else contagem_piscadas, quebrando a linha para facilitar a visualização.

```python
if ear < ear_limiar and mar < mar_limiar:
    t_inicial = time.time() if dormindo == 0 else t_inicial
    contagem_piscadas = contagem_piscadas+1 if dormindo == 0 else contagem_piscadas
```

## Cálculo de Frequêcia

Até aqui, já sabemos como contar as nossas piscadas. Agora, precisamos calcular a frequência em minutos em que essas piscadas acontecem. Sabemos que um minuto tem 60 segundos. Por isso, é interessante para o código armazenarmos a quantidade de piscadas que uma pessoa deu por segundo, armazenando assim 60 valores de piscadas nesse período. Assim, conseguiremos calcular a nossa média de piscadas por minuto. <br>

Para fazer isso, precisaremos usar o tempo da __biblioteca Time__. Indo no começo do nosso código setaremos a primeira variável para essa aplicação, a `t_piscadas`, ou o tempo para o cálculo de piscadas que faremos. Essa variável será igual a `time.time()`.

### Mas por que setar o tempo antes do início da análise?

Lembrando que precisamos calcular a quantidade de piscadas por minuto. Para fazer isso, precisamos ter um comparativo com um tempo inicial de referência para entender quantos minutos ou segundos se passaram. Então, dentro do bloco condicional `try`, após o `t_final = time.time()` vamos inserir uma nova varáivel `tempo_corrido`.  <br> Essa variável será igual a `t_final`, subtraída da `t_piscadas`. Assim, teremos o tempo decorrido entre o início da aplicação e o seu tempo atual. Com esse tempo decorrido, conseguimos marcar o número de piscadas de segundo em segundo.

```python
t_final = time.time()
tempo_decorrido = t_final - t_piscadas
```

Agora para termos essa marcação, voltaremos ao topo do nosso código e criaremos uma variável temporária chamada `c_tempo`, que contará o tempo de forma temporária. Seu valor será zero. Essa variável será útil na verificação de segundos porque, após criarmos o `tempo_decorrido`, podemos criar uma condicional `if` para determinar que ele execute algum trabalho se o `tempo_decorrido` for maior ou igual ao `c_tempo+1`. <br>

Se isso acontecer na primeira parte, usaremos a variável `tempo_decorrido` e verificaremos se ele é maior que um segundo. Nesse caso, atribuiremos o valor `tempo_decorrido` à variável `c_tempo`. Assim, sempre que o `if` acontecer, um piscada será coletada sempre um segundo após a última piscada coletada.

```python
if tempo_decorrido >= (c_tempo+1):
    c_tempo = tempo_decorrido
```

Feito isso, agora verificaremos quantas piscadas foram dadas dentro desse intervalo de um segundo de espera. Para isso, criaremos outra variável temporária no início do código, chamada `contagem_temporaria`, cujo valor será zero. Esta variável também se alterará conforme a última quantidade de piscadas antes de completarmos o intervalo de um segundo. Ela nos ajudará a contar quantas piscadas foram dadas por segundo. <br>

De volta à condicional `if`, criaremos uma variável chamada `piscadas_ps`, representando as piscadas por segundo. Ela será igual a `contagem_piscadas`, subtraída da `contagem_temporaria`. Isso nos permitirá contar as piscadas por minuto. Depois disso, podemos setar a `contagem_temporaria` igual a `contagem_piscadas`. Com isso, já podemos adicionar a quantidade de piscadas por segundo a uma lista que receberá esse número durante o período de 60 segundos.

```python
if tempo_decorrido >= (c_tempo+1):
    c_tempo = tempo_decorrido
    piscadas_ps = contagem_piscadas-contagem_temporaria
    contagem_temporaria = contagem_piscadas
```

Voltamos ao topo do código e criamos a nossa lista de contagem, chamada `contagem_lista`, que será igual a uma lista vazia []. Voltando para a condicional, adicionaremos o valor de piscadas por segundo na nossa variável `contagem_lista`, escrevendo `contagem_lista.append(piscadas_ps)`. Porém, não podemos ficar adicionando elementos infinitamente nessa lista sem uma limitação de quantidade de elementos. Como o intervalo para definir o número de piscadas por segundo é de 60 segundos, podemos limitar o tamanho de `contagem_lista` a 60 unidades. <br>

Logo abaixo, escreveremos: `contagem_lista = contagem_lista if (len(contagem_lista)<=60)`. Assim, se a contagem for menor ou igual a 60, ela continuará pertencendo a `contagem_lista`. Se essa condição for falsa, o código cortará e selecionará a lista. Essa alteração será indicada com o código: `else contagem_lista[-60:]`. Isso significa que a `contagem_lista` selecionará apenas os últimos 60 elementos, ou seja, do último elemento da lista até o elemento na posição -60.

```python
if tempo_decorrido >= (c_tempo+1):
    c_tempo = tempo_decorrido
    piscadas_ps = contagem_piscadas-contagem_temporaria
    contagem_temporaria = contagem_piscadas
    contagem_lista.append(piscadas_ps)
    contagem_lista = contagem_lista if (len(contagem_lista)<=60) else contagem_lista[-60:]
```

### Calculando as piscadas por minuto

Agora, podemos calcular as piscadas por minuto. Para isso, criaremos uma variável chamada `piscadas_pm` e podemos dizer que ela será igual ao somatório da nossa `contagem_lista`, ou seja `sum(contagem_lista)`. Com isso, serão contadas todas as piscadas por segundo dadas durante 60 segundos. <br>

Mas se não tiverem passado 60 segundos, `piscadas_pm` terá que adotar outro valor. Isso porque, se não se passaram 60 segundos, pode ser que tenhamos menos de 10 piscadas por segundo. Isso pode prejudicar a análise. Com isso, atribuiremos o valor 15 a `piscadas_pm` se o `tempo_decorrido` for menor ou igual a 60 segundos. Do contrário, a soma das contagens deve acontecer.

```python
piscadas_pm = 15 if tempo_decorrido<=60 else sum(contagem_lista)
```
