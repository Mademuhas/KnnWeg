from django.shortcuts import render
from django.http import HttpResponse
import json
import numpy as np

from django.views.decorators.csrf import csrf_exempt

import faiss
from sentence_transformers import SentenceTransformer


@csrf_exempt
def coment(request):
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    documentos = body['documents']
    bases = body['bases']


    textos = [documento['text'] for documento in documentos]
    titulos = [documento['subtitle'] for documento in documentos]
    base = [base['text'] for base in bases]
    output_base = [base['comment'] for base in bases]

    concat_amostras = [f"{texto}:{titulo}" for texto, titulo in zip(textos, titulos)]

    model = SentenceTransformer('intfloat/multilingual-e5-large')
    lista_knn = []
    lista_knn_total = []
    lista_embedding = []

    for i in range (len(base)):
        embedding_pm = model.encode(f'{base[i]}')
        lista_embedding.append(embedding_pm)

    index = faiss.IndexFlatL2(len(lista_embedding[0]))
    index.is_trained
    matrix = np.zeros ( (len(lista_embedding), len(lista_embedding[0]) ) )
    for i in range (matrix.shape[0]):
        matrix[i][:] = lista_embedding[i]
    index.add(matrix)
    
    k = 5

    for i in range (len(concat_amostras)):
        texto = f'{concat_amostras[i]}'
        xq = model.encode([texto])
        D, I = index.search(xq, k)
        lista_knn = I[0]
        lista_knn_total.append(lista_knn)
        lista_knn = []
    
    comentarios = []
    for i, knn in enumerate(lista_knn_total):
        for j in range(5):
            indice = lista_knn_total[i][j]
            comentarios.append(output_base[indice])
    count = 0
    lista_i = []
    for i in documentos:
        i['comment_1'] = comentarios[count]
        i['comment_2'] = comentarios[count+1]
        i['comment_3'] = comentarios[count+2]
        i['comment_4'] = comentarios[count+3]
        i['comment_5'] = comentarios[count+4]
        aux = json.dumps(i)
        aux = aux + ','
        lista_i.append(aux)
        count += 5

    return (HttpResponse(lista_i))





@csrf_exempt
def comply(request):
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    documentos = body['documents']
    bases = body['bases']


    textos = [documento['text'] for documento in documentos]
    titulos = [documento['subtitle'] for documento in documentos]
    base = [base['text'] for base in bases]
    output_base = [base['comply'] for base in bases]
    output_amostra = []

    concat_amostras = [f"{texto}:{titulo}" for texto, titulo in zip(textos, titulos)]

    model = SentenceTransformer('intfloat/multilingual-e5-large')
    lista_knn = []
    lista_knn_total = []
    lista_embedding = []



    for i in range (len(base)):
        embedding_pm = model.encode(f'{base[i]}')
        lista_embedding.append(embedding_pm)
        
        
    index = faiss.IndexFlatL2(len(lista_embedding[0]))
    index.is_trained
    matrix = np.zeros ( (len(lista_embedding), len(lista_embedding[0]) ) )
    for i in range (matrix.shape[0]):
        matrix[i][:] = lista_embedding[i]
    index.add(matrix)
    
    k = 5
 
    for i in range (len(concat_amostras)):
        texto = f'{concat_amostras[i]}'
        xq = model.encode([texto])
        D, I = index.search(xq, k)
        lista_knn = I[0]
        lista_knn_total.append(lista_knn)
        lista_knn = []


    for i in range (len(lista_knn_total)):
        count_ok = 0
        count_c = 0
        count_d = 0
        for j in range(5):
            indice = lista_knn_total[i][j]
            if output_base[indice] == 'OK':
                count_ok +=1
            elif output_base[indice] == 'C':
                count_c +=1
            elif output_base[indice] == 'D':
                if j < 2:
                    count_d += 100
                else:
                    count_d +=1

            
        if (count_ok > count_c) and (count_ok >= count_d):
            output_amostra.append('OK')
        elif (count_c >= count_ok) and (count_c > count_d):
            output_amostra.append('C')
        elif (count_c > count_ok) and (count_c >= count_d):
            output_amostra.append('C')
        elif (count_d > count_ok) and (count_d > count_c):
            output_amostra.append('D')

    cont_aux = 0
    lista_i = []
    for i in documentos:
        i['comply'] = output_amostra[cont_aux]
        cont_aux = cont_aux + 1
        aux = json.dumps(i)
        aux = aux+','
        lista_i.append(aux)
    

    return (HttpResponse(lista_i))

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")