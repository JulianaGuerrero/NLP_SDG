# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:00:42 2023

@author: Juliana Guerrero

Nuevo analisis de echo
"""
#pip install -U sentence-transformers
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import nltk
from nltk.corpus import stopwords
import re
import string
import numpy as np

# directorio 
os.chdir(r'C:\Users\ljuli\Desktop\UNFPA\SDGtrans')
# setting de mostrar todas las columnas
pd.set_option('display.max_columns', None)




# Download model
model3 = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# The sentences we'd like to encode
# lectura textos ods
obj = pd.read_csv(r'obj_240323.csv',encoding='latin-1', sep=',')
# concatenar campo final: texto original + temrinos
obj['tx_t'] = obj['objetivo']+obj['terminos']
# lectura de metas
metas = pd.read_csv(r'met_240323.csv',encoding='latin-1', sep=',')
# concatenar campo final: texto original + temrinos
metas['tx_t'] = metas['meta texto']+ metas['terminos']


## cleaning function
stop_words = stopwords.words("spanish")
def tx_clean(x):
    x = x.lower() #normalize
    x = ' '.join([word for word in x.split(' ') if word not in stop_words]) # remove stopwords
    x = re.sub(r'https*\S+', ' ', x) # remove links
    x = re.sub(r'@\S+', ' ', x) # remove mentions
    x = re.sub(r'#\S+', ' ', x) # remove hashtags
    x = re.sub(r'\'\w+', '', x) # remove ticks
    x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x) # remove punctuation
    x = re.sub(r'\w*\d+\w*', '', x) # remove numbers
    x = re.sub(r'\s{2,}', ' ', x) # remove extra spaces
    return x

# ejemplo en uno
#tx_clean('Desarrollar infraestructuras fiables,  sostenibles 1234 resilientes y de #calidad  incluidas infraestructuras regionales y transfronterizas  para apoyar el desarrollo económico y el bienestar humano  haciendo especial hincapié en el acceso asequible y equitativo para todos;infraestructura   carreteras  vías calle transporte   equipamientos  espacio público movilidad  puente señalización vías  ciclorutas  carrera avenida vías primarias vías terciarias pavimento asfalto huecos pavimentación')

# aplicar en toda la columna de objetivos y metas
obj['clean_text'] = obj.tx_t.apply(tx_clean)
metas['clean_text'] = metas.tx_t.apply(tx_clean)

# Get embeddings objetivos
obj_emb = model3.encode(obj.clean_text)
texto = 'Derecho a la alimentación : En temas de derecho a la alimentación, las personas que participaron en el DRV del Catatumbo establecen la importancia impulsar proyectos productivos, iniciativas económicas y acceso a tierra que incluyan mesas de participación intersectoriales entre el estado, la comunidad y las empresas presentes en el territorio, con un acompañamiento técnico para los proyectos de producción limpia y la sostenibilidad. Las propuestas recibidas estén representadas en transformación productiva con  48%; acceso Físico a alimentos con 38% y alimentos sanos y seguros con un14% Así mismo, plantean que la región del Catatumbo inicie la transición de una economía basada en el extractivismo, la economía de uso ilícito y monocultivos  a un sistema agroalimentario que garantice la soberanía y seguridad alimentaria local, fundamentado en el impulso de la agroecología. Por otra parte, dan importancia a la implementación del Plan de Desarrollo Agrícola'
# crear funcion 
def sdg(texto):
    # Compute similarities objetivos y texto
    sim_list_obj = util.cos_sim(obj_emb,model3.encode(texto)).tolist()
    # index more similar obj
    obj_id = np.argsort(sim_list_obj,axis=0)[::-1][:3]
    # metas
    df_metas = pd.DataFrame()
    for x in obj_id+1:
        # filtrar objetivo
        temp_df = metas[metas.id_objetivo==x.item()]
        #append df
        df_metas= df_metas.append(temp_df)
        df_metas = df_metas.reset_index(drop=True)
    # embedding
    met_emb = model3.encode(df_metas.clean_text)    
    # similarity
    met_sim =  util.cos_sim(met_emb,model3.encode(texto)).tolist()
    met_id = np.argsort(met_sim,axis=0)[::-1][:5]    
    # salidas
    sal_meta = df_metas.filter(met_id[:,0].tolist(),axis=0)
    sal_obj = obj.filter(obj_id[:,0].tolist(),axis=0)
    return(sal_obj['objtot'],sal_meta['met_conc'])


# salida con un texto ejemplo
sdg('En cierre de brechas sociales y económicas las propuestas con mayor interés están proyectadas a educación, con un 54%. Acá la comunidad relaciona este sector como un eje que debería incluir estrategias metodologías, de inversión en infraestructura educativa y vial; legalización de predios para escuelas; garantías de acceso a nuevas tecnologías; enfoques de proyectos educativos institucionales; estabilidad de los docentes y acceso a alimentos de calidad durante la jornada escolar principalmente. Todas estas direccionados hacia competencias rurales. Con un enfoque poblacional para jóvenes y mujeres.')

sdg('Capacidad InstitucionalLa comunidad relaciona con la capacidad institucional con voluntad política, presencia en territorio de entidades públicas, acciones transparentes que eviten la corrupción, cumplimiento de acciones judiciales y el diálogo permanente entre el gobierno y comunidad. Aspectos que articulados permitan identificar al Catatumbo como una zona especial transformadora de la región, donde la integración territorial sea base de la transformación.')
sdg('invasión espacio público recolección residuos negocios comerciales invadido andenes permitir paso peatones además sacan residuos cumplir horarios recolección estacionamiento vehículos vías generando trancones además vías encuentran pésimo')

sdg('Derecho a la alimentación : En temas de derecho a la alimentación, las personas que participaron en el DRV del Catatumbo establecen la importancia impulsar proyectos productivos, iniciativas económicas y acceso a tierra que incluyan mesas de participación intersectoriales entre el estado, la comunidad y las empresas presentes en el territorio, con un acompañamiento técnico para los proyectos de producción limpia y la sostenibilidad. Las propuestas recibidas estén representadas en transformación productiva con 48%; acceso Físico a alimentos con 38% y alimentos sanos y seguros con un 14% Así mismo, plantean que la región del Catatumbo inicie la transición de una economía basada en el extractivismo, la economía de uso ilícito y monocultivos a un sistema agroalimentario que garantice la soberanía y seguridad alimentaria local, fundamentado en el impulso de la agroecología. Por otra parte, dan importancia a la implementación del Plan de Desarrollo Agrícola')

sdg('Paz, Defensa y Seguridad: Al igual que en la capacidad institucional, la voluntad política es un factor que determina las propuestas en esta variable de análisis, en lo particular direccionadas al cumplimiento de los Acuerdos de Paz con un 48%, las relacionadas con cultivos ilícitos le siguen con un 36% y una tercera prioridad con un 14% la protección a la vida humana. En un 9 % se ubican propuestas relacionadas con reparación a víctimas y seguridad.  Conflicto y grupos armados al margen de la ley, junto con fronteras se posicionan con un 7%. Finalmente, las propuestas en desplazamiento forzado se representan en un 2%. Para la paz, buscan el reconocimiento de la “mesa humanitaria de construcción de Paz del Catatumbo como mecanismo de participación y articulación social')

sdg('vereda colorada ubicada frente estación servicio colina vía arcabuco aparece documentos vereda pirgua vereda colorada ahora documentos aparece urbano rural serí además vía principal encuentra mal debería dejar pot urbano cuenta anteriormente dicho')

sdg('parece seria buena idea destinar horarios recolección basura doy cuenta cumpliendo tare reciclar parecería bien ubiesen almenos horarios diferentes recoger basura decir dia reciclaje día desechable si ed')

sdg('principalmente carencia servicios complementarios deterioro malla vial ningun componente urbano permita movilizarse pie bici barrio vivo grandes razón bueno complementara servicios complementarios igual garantizar buena malla vial estructuras peatonales sistemas movilidad diferentes tradicionales')

sdg('pocas zonas verdes utilizadas sitios perros hagan necesidades congestión vial usarse vías residenciales vias alternas alta congestión vehicular camara comercio planea actividades horario nocturno genera molestias')

sdg('relleno sanitario olores invasión bichos perros comen animales hornos chircales horas día afectación humo falta tecnificación vías medios transporte falta vivienda falta subisidio vivienda solución predios castratales subdivisio')

sdg('sector importante definir claramente uso suelo defina tema actividades permitidas especial relacionado zona tolerancia clara q existe tema conflicto sector carrera cerca antiguo terminal frente trabajadoras sexuales deben')

sdg('mejor señalización tránsito educación normas tránsito comunidad construcción puente quintas debe cambiar estructura vías pasarla adoquín vías flexibles alto tráfico impmemtacion semáforos mantenimiento reposición parques infanti')

sdg('barrio necesario fortalecer tema privatización lotes si bien cierto dueño dejan abiertos fomenta inseguridad desaseo barrio necesario fortalecer tema privatización lotes si bien cierto dueño dejan abiertos fomenta inseguridad desaseo ademas necesario regular recolección material reciclar personas encargadas rompen')

sdg('garantizar dentro clasificación equipamientos colectivos sociales queden registradas iglesias espacios culto garantizar condiciones edificabilidad sector religioso índice ocupación igual mínimo índice construcción sector religioso')

sdg('falta políticas parte alcaldía hacer cultura ciudadana mejor ordenamiento territorial cuanto definir mesopotamia únicamente residencial debido hacía carreta sexta bodegas cemento entran salen vehículo carga pesada dañan malla vial f')

sdg('salida hacia arcabuco clínica veterinaria uptc salida bomba colina andenes vía sectores iluminación si observamos vía salida hacia arcabuco puerta bienvenida sido descuidada mantenimiento vía sectores andenes vía luz pública haciendo foco inseguridad alto riesgo transeúntes')

sdg('legalizacion barrio barrio aparece sistena municipio legalizado documentos aparece cedido municipio alcantarillado obsoleto vias totalidad presentan mal sicavación via vecina barrio adhiere afectación mismo')

sdg('control desarrollo vivienda vías crecimiento vivienda desordenado parece coherencia contexto enfoque mejora convirtiendo zonas duras calidad apego medio ambiente cada persona hace vivienda quiere tener cuenta manejo lectur')

sdg('sector antiguo terminal zona central realmente existe mucha delincuencia zona tolerancia prostitución afecta imagen además afecta crecimiento económico sector zonas tolerancia deberían ser reubicadas s sector antiguo terminal zona central realmente existe mucha delincuencia zona tolerancia prostitución afecta imagen además afecta crecimiento económico sector zonas tolerancia deberían ser reubicadas s')
