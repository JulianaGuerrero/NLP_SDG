# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:17:56 2021

@author: Juliana Guerrero Velasquez

Similitud objetivos de desarrollo sostenible 
"""
# librerias 
import os
import contexto
from contexto.lectura import Lector, leer_texto
from contexto.escritura import Escritor, escribir_texto
from contexto.limpieza import *
from contexto.comparacion import Similitud, Distancia, DiferenciaStrings
from contexto.vectorizacion import *
import nltk
from collections import Counter   
# directorio 
os.chdir(r'C:\Users\ljuli\Desktop\UNFPA\ECHO\modelo_similitud')
# setting de mostrar todas las columnas
pd.set_option('display.max_columns', None)
# lectura textos ods
obj = pd.read_csv(r'input\objetivos_texto_v3.csv',encoding='latin-1', sep=';')
# guardar terminos
objetivos = obj.terminos

# palabras vacías
stopwords = lista_stopwords()
# limpieza textos objetivos
objetivos_l = [limpieza_texto(item,lista_palabras=stopwords) for item in objetivos]


## Definir vectorizador
v_word2vec = VectorizadorWord2Vec()
## vectorizar objetivos
vectores = v_word2vec.vectorizar(objetivos_l)    
# Inicializar objetos de clase Similitud
s_word2vec = Similitud(v_word2vec)

# Lectura respuestas
res = pd.read_csv(r'textos_medellin.csv',encoding='latin-1')
#res = pd.read_csv(r'input\txt_medellin_completo.csv',encoding='latin-1',sep=';')
#res = pd.read_csv(r'input/villavicencio_texto.csv',encoding='latin-1',sep=';')

##############################################################################
## objetivos
##############################################################################

# funcion de calculo similitud respuesta y objetivos
def comp(texto):
    coseno_word2vec = s_word2vec.coseno(objetivos_l, limpieza_texto(texto))
    obj_id = np.argsort(coseno_word2vec.flatten())[::-1][:3]
    obj_sim = sorted(coseno_word2vec.flatten(),reverse=True)[:3]
    return(obj_id,obj_sim)
# prueba
comp('ademas lleva años sido tratada calle visto deteriorada') # 8,1,10
# 5,3,1
# ajuste de similitud
salida = res.respuesta.apply(lambda x: comp(x))

# id objetivos
res['objetivo_1'] = [item[0][0] for item in salida]
res['objetivo_2'] = [item[0][1] for item in salida]
res['objetivo_3'] = [item[0][2] for item in salida]
# similitud objetivos
res['objetivo_1_sim'] = [item[1][0] for item in salida]
res['objetivo_2_sim'] = [item[1][1] for item in salida]
res['objetivo_3_sim'] = [item[1][2] for item in salida]
# arreglar ids-llaves objetivos 
res['objetivo_1'] =res['objetivo_1'] +1
res['objetivo_2'] =res['objetivo_2'] +1
res['objetivo_3'] =res['objetivo_3'] +1

##############################################################################
## metas
##############################################################################

metas = pd.read_csv(r'input\metas_texto_v3.csv',encoding='latin-1', sep=';')

# funcion similitud con metas
def met_comp(texto_meta,texto):
    coseno_word2vec = s_word2vec.coseno(texto_meta, limpieza_texto(texto))
    met_id = np.argsort(coseno_word2vec.flatten())[::-1][0]
    met_sim = sorted(coseno_word2vec.flatten(),reverse=True)[0]
    return(met_id,met_sim)

met_comp('falta mucha contratación generar empleo')
#  meta dentro del primer objetivo
df_obj1_final = pd.DataFrame()
for i in range(1,18):
    df = pd.DataFrame()
    # preparacion textos
    met_df = metas[metas.id_objetivo==i]['terminos']
    texto_meta = [limpieza_texto(item,lista_palabras=stopwords) for item in met_df]
    res_df = res[res.objetivo_1==i]['respuesta']
    if len(res_df)!=0:
            # comparacion textos
            sal = res_df.apply(lambda x: met_comp(texto_meta,x))
            # salida
            df['meta_1_id'] =  [item[0] for item in sal]
            df['meta_1_sim'] =  [item[1] for item in sal]
            df['id'] = sal.index
            df['objetivo'] = i
            df_obj1_final = df_obj1_final.append(df)

df_obj1_final = df_obj1_final.sort_values('id')
# pegar salida primer meta
pr1 = pd.concat([res.reset_index(drop=True),df_obj1_final[['meta_1_id','meta_1_sim']].reset_index(drop=True)],axis=1)

#  meta dentro del segundo objetivo
df_obj2_final = pd.DataFrame()
for i in range(1,18):
    df = pd.DataFrame()
    # preparacion textos
    met_df = metas[metas.id_objetivo==i]['terminos']
    texto_meta = [limpieza_texto(item,lista_palabras=stopwords) for item in met_df]
    res_df = res[res.objetivo_2==i]['respuesta']
    if len(res_df)!=0:
            # comparacion textos
            sal = res_df.apply(lambda x: met_comp(texto_meta,x))
            # salida
            df['meta_2_id'] =  [item[0] for item in sal]
            df['meta_2_sim'] =  [item[1] for item in sal]
            df['id'] = sal.index
            df['objetivo'] = i
            df_obj2_final = df_obj2_final.append(df)

df_obj2_final = df_obj2_final.sort_values('id')

#  meta dentro del tercer objetivo
pr1 = pd.concat([pr1.reset_index(drop=True),df_obj2_final[['meta_2_id','meta_2_sim']].reset_index(drop=True)],axis=1)

# tercera meta
df_obj3_final = pd.DataFrame()
for i in range(1,18):
    df = pd.DataFrame()
    # preparacion textos
    #i=3
    met_df = metas[metas.id_objetivo==i]['terminos']
    texto_meta = [limpieza_texto(item,lista_palabras=stopwords) for item in met_df]
    res_df = res[res.objetivo_3==i]['respuesta']
    if len(res_df)!=0:
            # comparacion textos
            sal = res_df.apply(lambda x: met_comp(texto_meta,x))
            # salida
            df['meta_3_id'] =  [item[0] for item in sal]
            df['meta_3_sim'] =  [item[1] for item in sal]
            df['id'] = sal.index
            df['objetivo'] = i
            df_obj3_final = df_obj3_final.append(df)

df_obj3_final = df_obj3_final.sort_values('id')

# pegar salida tercer meta
pr1 = pd.concat([pr1.reset_index(drop=True),df_obj3_final[['meta_3_id','meta_3_sim']].reset_index(drop=True)],axis=1)

# re-enumerar 
pr1['meta_1_id'] = pr1['meta_1_id']+1
pr1['meta_2_id'] = pr1['meta_2_id']+1
pr1['meta_3_id'] = pr1['meta_3_id']+1

# guarda salida 
pr1.to_csv('output/medellin_comp_221021.csv',index=False,sep=';')
