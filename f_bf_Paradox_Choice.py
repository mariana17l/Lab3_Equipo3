#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 07:58:28 2019

@author: Equipo C
"""

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go # Si no se puede con plotly.graph_objects trate plotly.graph_objs
from plotly.offline import init_notebook_mode, plot
from datetime import datetime , timedelta
init_notebook_mode(connected=True)
import json
#%% Opciones para visualizar data frames en consola
pd.set_option("display.max_rows",5000)
pd.set_option("display.max_columns",500)
pd.set_option("display.width",1000)
#style.use("ggplot")


#%%
def f_bf_Paradox_Choice(dataframe):
    """
    :param dataframe: Excel del histórico del trader
    
    :return:  Diccionario con el resultado final con 4 elementos.
              "datos":  tabla_explicativa -> Número de activos utilizados durante cierto periodo donde distinguimos entre positivos y negativos.
                        activos_deseados -> Activos en los que el inversionista debería centrar su atención.
              "grafica": 4 graficas representando periodos diarios y semanales (positivos y negativos).
              "escala": El porcentaje del número de activos que el trader debería utilizar del total.
              "explicacion":  Breve explicación del sesgo."""

    # Definir funciones
    
    def choose_index_values(dataframe,l):
        a = dataframes[l].iloc[:,[3,12]].groupby("Symbol").sum()
        #b = [dataframes[l].iloc[:,[3,12]].groupby("Symbol").sum()>0]
        c = a[a>0].dropna()
        index = c.index.values.tolist()
        values = c.values.tolist()
        return index, values
    
    def choose_index_values_2(dataframe,l):
        a = dataframes[l].iloc[:,[3,12]].groupby("Symbol").sum()
        #b = [dataframes[l].iloc[:,[3,12]].groupby("Symbol").sum()<0]
        c = a[a<0].dropna()
        index = c.index.values.tolist()
        values = c.values.tolist()
        return index, values

    
    def subplots_positivos_negativos(profit_loss,positivo =True, day =True):
        # Para saber si son pérdidas o ganancias y si es diario o semanal
        if positivo:
            utilizar = ["rendimiento", "son positivos", "Ganancia"]
        else:
            utilizar = ["pérdida", "no son positivos", "Pérdida"]

        if day:
            periodo = ["Días"]
        else:
            periodo = ["Semanas"]

        fig = make_subplots(
            rows=2, cols=2, subplot_titles=("No. Activos diarios que obtuvieron "+ utilizar[0],
                                            "Frecuencia de Activos Utilizados cuando " + utilizar[1],
                                            utilizar[2]+ " Promedio por "+str(periodo), "Frecuecia " +utilizar[2]+ " Promedio por "+str(periodo))
        )

        # Add traces
        fig.add_trace(go.Scatter( y=[len(profit_loss[l][0]) for l in range(len(profit_loss))]), row=1, col=1)
        fig.add_trace(go.Histogram( x=[len(profit_loss[l][0]) for l in range(len(profit_loss))],
                xbins=dict(size=0.3)), row=1, col=2)
        fig.add_trace(go.Scatter( y=[np.mean(profit_loss[l][1]) for l in range(len(profit_loss)) if len(profit_loss[l][1])>0]), row=2, col=1)
        fig.add_trace(go.Histogram(x=[np.mean(profit_loss[l][1]) for l in range(len(profit_loss)) if len(profit_loss[l][1])>0],
                xbins=dict(size=4.5)), row=2, col=2)

        # Update xaxis properties
        fig.update_xaxes(title_text=periodo[0], row=1, col=1)
        fig.update_xaxes(title_text="No. Activos",row=1, col=2)
        fig.update_xaxes(title_text=periodo[0], row=2, col=1)
        fig.update_xaxes(title_text="Dinero", row=2, col=2)

        # Update yaxis properties
        fig.update_yaxes(title_text="No. Activos", row=1, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=1, col=2)
        fig.update_yaxes(title_text="Dinero",row=2, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=2, col=2)

        # Update title and height
        fig.layout.update(showlegend=False, title_text="Análisis de " + utilizar[2]+"s", height=700)

        #plot(fig,filename= 'basic-scatter')
       
        return
    
    
    # Ajustar en Index el openTime y borrarlo como columna
    data = dataframe.copy() # Este dataset se utilizará cuando se realice el análisis semanal.
    num_Activos = len(np.unique(data["Symbol"].values))
    dataframe.index = dataframe["openTime"]; dataframe = dataframe.drop("openTime",axis=1); 
    dataframe = dataframe.sort_index() 
    data_mensual = dataframe.copy()
    dataframe.index = [dataframe.index[k][:10] for k in range(dataframe.shape[0])]  # Quitando la hora del índice, dejando solo fecha : "yyy.mm.dd"
    dataframes = []  # lista vacia
    unicos = np.unique(dataframe.index)  # del índice se buscan las fechas únicas para clasificar la información de acuerdo a estas.
    
    for k in range(len(unicos)):
        dataframes.append(pd.DataFrame(dataframe.loc[dataframe.index==unicos[k],:])) #agregando a la lista los dataframes con la info separada.  
    
    # Conocer los potiviso y negativos diarios
    profits_day = [choose_index_values(dataframes,l) for l in range(len(dataframes)) if len(choose_index_values(dataframes,l)[0]) ]
    no_profits_day = [choose_index_values_2(dataframes,l) for l in range(len(dataframes)) if len(choose_index_values_2(dataframes,l)[0])>=1]
    # Para rellenar tabla
    num_Activos_ganancia_day = len(np.unique(np.concatenate([l[0] for l in profits_day])))
    num_Activos_perdida_day = len(np.unique(np.concatenate([l[0] for l in no_profits_day])))
    # Graficar
    diario_positivo = subplots_positivos_negativos(profit_loss=profits_day,positivo =True, day =True)
    diario_negativo = subplots_positivos_negativos(profit_loss=no_profits_day,positivo =False, day =True)
    
    # Análisis semanal
    data2= data
    data2["'openTime'"] = [data2['openTime'][k][:10] for k in range(data2.shape[0])]
    data2['openTime'] = [datetime.strptime(data2["'openTime'"][k], '%Y.%m.%d') for k in range(data2.shape[0])]
    dty=[]
    gr = data2.groupby(pd.Grouper(key='openTime',freq='W'))
    for name, group in gr:
        if len(group) > 0:        
            dty.append(group)   
    dty=[pd.DataFrame(dty[i]).set_index('openTime') for i in range(len(dty))]
    dty =[l.drop("'openTime'",axis=1) for l in  dty]
    dataframes=dty
    # Conocer los positivos y Negativos semanles
    profits_week = [choose_index_values(dataframes,l) for l in range(len(dataframes)) if len(choose_index_values(dataframes,l)[0]) ]
    no_profits_week = [choose_index_values_2(dataframes,l) for l in range(len(dataframes)) if len(choose_index_values_2(dataframes,l)[0])>=1]
    # Para rellanar tabla
    num_Activos_ganancia_week = len(np.unique(np.concatenate([l[0] for l in profits_week])))
    num_Activos_perdida_week = len(np.unique(np.concatenate([l[0] for l in no_profits_week])))
    # Graficar
    semanal_positivo = subplots_positivos_negativos(profit_loss=profits_week,positivo =True, day =False)
    semanal_negativo = subplots_positivos_negativos(profit_loss=no_profits_week,positivo =False, day =False)
    
    # Realizar grafica que explique todo lo que las graficas muestran
    tabla_explicativa = pd.DataFrame(columns = ["No. de Activos \n utilizados en total","No. de Activos que \n solo producieron pérdida",
                                               "No. de Activos que \n solo producieron ganancia"],index = ["Diario","Semanal","Mensual"])
    tabla_explicativa.loc["Diario",:] = [num_Activos,num_Activos_perdida_day,num_Activos_ganancia_day]
    tabla_explicativa.loc["Semanal",:] = [num_Activos,num_Activos_perdida_week,num_Activos_ganancia_week]
    num_Activos_ganancia_month = pd.DataFrame(data.groupby(["Symbol"]).sum()["Profit"][data.groupby(["Symbol"]).sum()["Profit"]>0])
    num_Activos_perdida_month = pd.DataFrame(data.groupby(["Symbol"]).sum()["Profit"][data.groupby(["Symbol"]).sum()["Profit"]<0])
    tabla_explicativa.loc["Mensual",:] = [num_Activos,num_Activos_perdida_month.shape[0],num_Activos_ganancia_month.shape[0]]
    
    
    # Activos en lo que se recomienda invertir
    data = data_mensual
    # Se toman los activos mensuales con rendimiento positivo, se ordenan de mayor a menor y se juntan en un solo vector
    concatenados = np.concatenate(np.sort(pd.DataFrame(data.groupby(["Symbol"]).sum()["Profit"][data.groupby(["Symbol"]).sum()["Profit"]>0]).values))
    # Como queremos saber cuantos activos explican por lo menos el 95% del rendimiento 
    # quitamos aquellos que resultan insignificativos
    ultimo_significativo = (np.sort(concatenados)[::-1])[np.cumsum(np.sort(concatenados)[::-1])/np.sum(concatenados)<=.95][-1]
    activos_deseados = num_Activos_ganancia_month[num_Activos_ganancia_month.values>=ultimo_significativo].dropna()
    graficas = {"DiarioP":diario_positivo,"DiarioN":diario_negativo,"SemanalP":semanal_positivo,"SemanalN":semanal_negativo}
    escalas = {"escala":np.round(1- (abs(-num_Activos+len(activos_deseados.values)))/num_Activos,3), "text":" % de activos en los que el trader debería enfocarse del total."}
    return {"datos":[tabla_explicativa, activos_deseados],"grafica":graficas,"escala":escalas,
            "explicacion": """Paradox Choice: Sesgo donde se argumenta que mientras más elecciones se tiene más ansiedad se provoca, por lo tanto, disminuir nuestra posibilidad de elección puede resultar mejor. En el área de Trading (en nuestro análisis) lo que se busca en reducir el número de instrumentos con los que se opera y que explican la mayor parte de tu rendimiento."""}