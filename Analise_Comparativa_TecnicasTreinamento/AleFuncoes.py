import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def unicos_qtds(vetor):
    valores = []
    qtds = []
    qtd_unicos = 0
    v = np.nan
    qtd = 0
    for u in np.sort(vetor):
        if u == v:
            qtd = qtd + 1
        else:
            valores.append(v)
            qtds.append(qtd)
            qtd_unicos = qtd_unicos + 1
            v = u
            qtd = 1
    valores.append(v)
    qtds.append(qtd)
    return np.array(valores[1:]), np.array(qtds[1:]), qtd_unicos

def pontos_corte(qtds, qtd_unicos, num_div):
    passo = int(qtd_unicos/num_div)
    pts_corte = [i*passo for i in range(num_div+1)]
    pts_corte[-1] = qtd_unicos
    qtds_corte = [np.sum(qtds[pts_corte[i]:pts_corte[i+1]]) for i in range(num_div)]
    pts_corte = [p - 1 for p in pts_corte[1:]]
    return np.array(pts_corte), np.array(qtds_corte)

def pontos_corte(qtds, qtd_unicos, num_div):
    passo = int(qtd_unicos/num_div)
    pts_corte = [i*passo for i in range(num_div+1)]
    pts_corte[-1] = qtd_unicos
    qtds_corte = [np.sum(qtds[pts_corte[i]:pts_corte[i+1]]) for i in range(num_div)]
    pts_corte = [p - 1 for p in pts_corte[1:]]
    return np.array(pts_corte), np.array(qtds_corte)

def minimiza_desvio_padrao(pts_corte, qtds_corte, qtds, qtd_unicos):
    fim = pts_corte.size - 1
    prefim = fim - 1
    permutado = True
    while(permutado):
        permutado = False
        for i in range(fim):
            a = qtds_corte[prefim-i]
            b = qtds_corte[fim-i]
            p = pts_corte[prefim-i]
            qe = qtds[p]
            qd = qtds[p+1]
            if(a - b > qe and a > qe):
                qtds_corte[fim-i] = b + qe
                qtds_corte[prefim-i] = a - qe
                pts_corte[prefim-i] = p - 1
                permutado = True
            elif(b - a > qd and b > qd):
                qtds_corte[fim-i] = b - qd
                qtds_corte[prefim-i] = a + qd
                pts_corte[prefim-i] = p + 1
                permutado = True
    return pts_corte, qtds_corte

def calcula_valores_corte(valores, pts_corte):
    valores_intermed = (valores[pts_corte[:-1]] + valores[pts_corte[:-1]+1])/2
    valores_min = np.append(valores[0], valores_intermed)
    valores_max =  np.append(valores_intermed, valores[pts_corte[-1]])
    return valores_min, valores_max

def calcula_min_diff(vetor):
    return np.min(np.diff(vetor))

def calcula_pontos_medios(valores_min, valores_max):
    return (valores_min + valores_max)/2

def calcula_pares_minmax(valores_min, valores_max):
    return np.array(list(zip(valores_min, valores_max)))

def conta_qtds_bin(inf_bin, sup_bin, valores_min, valores_max, qtds):
    qtd_bin = 0
    for i in range(qtds.size):
        if(inf_bin <= valores_min[i] and sup_bin >= valores_max[i]):
            qtd_bin = qtd_bin + qtds[i]
        elif(inf_bin <= valores_min[i] and sup_bin < valores_max[i] and sup_bin > valores_min[i]):
            qtd_bin = qtd_bin + qtds[i]*(sup_bin - valores_min[i])/(valores_max[i] - valores_min[i])
        elif(inf_bin > valores_min[i] and sup_bin >= valores_max[i] and inf_bin < valores_max[i]):
            qtd_bin = qtd_bin + qtds[i]*(valores_max[i] - inf_bin)/(valores_max[i] - valores_min[i])
        elif(inf_bin > valores_min[i] and sup_bin < valores_max[i]):
            qtd_bin = qtd_bin + qtds[i]*(sup_bin - inf_bin)/(valores_max[i] - valores_min[i])
    return qtd_bin

def discretiza_vetor(vetor, valores_digitize):
    flag_nan = np.isnan(vetor)
    disc = np.digitize(vetor, valores_digitize)
    return np.where(flag_nan, np.nan, disc)

def discretiza_vetor_media(vetor, valores_digitize, valores_medios):
    flag_nan = np.isnan(vetor)
    disc = np.digitize(vetor, valores_digitize)
    return np.where(flag_nan, np.nan, valores_medios[disc])

def unicos_qtds_alvos(vetor, alvo):
    inds_sorted = np.argsort(vetor)
    vetor_sorted = vetor[inds_sorted]
    alvo_sorted = alvo[inds_sorted]
    valores = []
    qtds = []
    qtds_alvo = []
    qtd_unicos = 0
    v = np.nan
    qtd = 0
    qtd_alvo = 0
    for i in range(vetor_sorted.size):
        if vetor_sorted[i] == v:
            qtd = qtd + 1
            qtd_alvo = qtd_alvo + alvo_sorted[i]
        else:
            valores.append(v)
            qtds.append(qtd)
            qtds_alvo.append(qtd_alvo)
            qtd_unicos = qtd_unicos + 1
            v = vetor_sorted[i]
            qtd = 1
            qtd_alvo = alvo_sorted[i]
    valores.append(v)
    qtds.append(qtd)
    qtds_alvo.append(qtd_alvo)
    return np.array(valores[1:]), np.array(qtds[1:]), np.array(qtds_alvo[1:]), qtd_unicos

def pontos_corte_alvo(qtds, qtd_unicos, qtds_alvo, num_div):
    passo = int(qtd_unicos/num_div)
    pts_corte = [i*passo for i in range(num_div+1)]
    pts_corte[-1] = qtd_unicos
    qtds_corte = [np.sum(qtds[pts_corte[i]:pts_corte[i+1]]) for i in range(num_div)]
    qtds_alvo_corte = [np.sum(qtds_alvo[pts_corte[i]:pts_corte[i+1]]) for i in range(num_div)]
    pts_corte = [p - 1 for p in pts_corte[1:]]
    return np.array(pts_corte), np.array(qtds_corte), np.array(qtds_alvo_corte)

def funcao_comparacao_entropia(Qe, Ae, Qd, Ad):
    retorno = Qe*np.log2(Qe) + Qd*np.log2(Qd)
    if(Ae > 0):
        retorno = retorno - Ae*np.log2(Ae)
    if(Ae < Qe):
        retorno = retorno - (Qe-Ae)*np.log2(Qe-Ae)
    if(Ad > 0):
        retorno = retorno - Ad*np.log2(Ad)
    if(Ad < Qd):
        retorno = retorno - (Qd-Ad)*np.log2(Qd-Ad)
    return retorno
    
def funcao_comparacao_entropia_balanceada(Qe, Ae, Qd, Ad):
    retorno = np.log2(Qe) + np.log2(Qd)
    if(Ae > 0):
        retorno = retorno - Ae*np.log2(Ae)/Qe
    if(Ae < Qe):
        retorno = retorno - (Qe-Ae)*np.log2(Qe-Ae)/Qe
    if(Ad > 0):
        retorno = retorno - Ad*np.log2(Ad)/Qd
    if(Ad < Qd):
        retorno = retorno - (Qd-Ad)*np.log2(Qd-Ad)/Qd
    return retorno

def minimiza_entropia(pts_corte, qtds_corte, qtds, qtds_alvo_corte, qtds_alvo, qtd_unicos, qtd_min):
    fim = pts_corte.size - 1
    prefim = fim - 1
    permutado = True
    while(permutado):
        permutado = False
        for i in range(fim):
            Qe = qtds_corte[prefim-i]
            Qd = qtds_corte[fim-i]
            Ae = qtds_alvo_corte[prefim-i]
            Ad = qtds_alvo_corte[fim-i]
            p = pts_corte[prefim-i]
            
            p_inf = 0 if i == prefim else pts_corte[prefim-i-1]
            e = p
            while(permutado == False and e > p_inf):
                qe = np.sum(qtds[e:p+1])
                ae = np.sum(qtds_alvo[e:p+1])
                if(Qe > qe + qtd_min and funcao_comparacao_entropia(Qe-qe, Ae-ae, Qd+qe, Ad+ae) < funcao_comparacao_entropia(Qe, Ae, Qd, Ad)):
                    qtds_corte[fim-i] = Qd + qe
                    qtds_corte[prefim-i] = Qe - qe
                    qtds_alvo_corte[fim-i] = Ad + ae
                    qtds_alvo_corte[prefim-i] = Ae - ae
                    pts_corte[prefim-i] = e - 1
                    permutado = True
                else:
                    e = e - 1
            
            p_sup = pts_corte[fim-i]
            d = p+1
            while(permutado == False and d < p_sup):
                qd = np.sum(qtds[p+1:d+1])
                ad = np.sum(qtds_alvo[p+1:d+1])
                if(Qd > qd + qtd_min and funcao_comparacao_entropia(Qe+qd, Ae+ad, Qd-qd, Ad-ad) < funcao_comparacao_entropia(Qe, Ae, Qd, Ad)):
                    qtds_corte[fim-i] = Qd - qd
                    qtds_corte[prefim-i] = Qe + qd
                    qtds_alvo_corte[fim-i] = Ad - ad
                    qtds_alvo_corte[prefim-i] = Ae + ad
                    pts_corte[prefim-i] = d
                    permutado = True
                else:
                    d = d + 1
                
    return pts_corte, qtds_corte, qtds_alvo_corte
    
def minimiza_entropia_balanceada(pts_corte, qtds_corte, qtds, qtds_alvo_corte, qtds_alvo, qtd_unicos, qtd_min):
    fim = pts_corte.size - 1
    prefim = fim - 1
    permutado = True
    while(permutado):
        permutado = False
        for i in range(fim):
            Qe = qtds_corte[prefim-i]
            Qd = qtds_corte[fim-i]
            Ae = qtds_alvo_corte[prefim-i]
            Ad = qtds_alvo_corte[fim-i]
            p = pts_corte[prefim-i]
            
            p_inf = 0 if i == prefim else pts_corte[prefim-i-1]
            e = p
            while(permutado == False and e > p_inf):
                qe = np.sum(qtds[e:p+1])
                ae = np.sum(qtds_alvo[e:p+1])
                if(Qe > qe + qtd_min and funcao_comparacao_entropia_balanceada(Qe-qe, Ae-ae, Qd+qe, Ad+ae) < funcao_comparacao_entropia_balanceada(Qe, Ae, Qd, Ad)):
                    qtds_corte[fim-i] = Qd + qe
                    qtds_corte[prefim-i] = Qe - qe
                    qtds_alvo_corte[fim-i] = Ad + ae
                    qtds_alvo_corte[prefim-i] = Ae - ae
                    pts_corte[prefim-i] = e - 1
                    permutado = True
                else:
                    e = e - 1
            
            p_sup = pts_corte[fim-i]
            d = p+1
            while(permutado == False and d < p_sup):
                qd = np.sum(qtds[p+1:d+1])
                ad = np.sum(qtds_alvo[p+1:d+1])
                if(Qd > qd + qtd_min and funcao_comparacao_entropia_balanceada(Qe+qd, Ae+ad, Qd-qd, Ad-ad) < funcao_comparacao_entropia_balanceada(Qe, Ae, Qd, Ad)):
                    qtds_corte[fim-i] = Qd - qd
                    qtds_corte[prefim-i] = Qe + qd
                    qtds_alvo_corte[fim-i] = Ad - ad
                    qtds_alvo_corte[prefim-i] = Ae + ad
                    pts_corte[prefim-i] = d
                    permutado = True
                else:
                    d = d + 1
                
    return pts_corte, qtds_corte, qtds_alvo_corte

def entropia_shannon(vetor_p1):
    entropia = []
    for p1 in vetor_p1:
        p0 = 1 - p1
        if p0 == 0 or p1 == 0:
            entropia.append(0)
        else:
            entropia.append(-p0*np.log2(p0) - p1*np.log2(p1))
    return np.array(entropia)

def indices_qtds(vetor):
    inds_sorted = np.argsort(vetor)
    vetor_sorted = vetor[inds_sorted]
    qtds = []
    qtd_unicos = 0
    v = np.nan
    qtd = 0
    for u in vetor_sorted:
        if u == v:
            qtd = qtd + 1
        else:
            qtds.append(qtd)
            qtd_unicos = qtd_unicos + 1
            v = u
            qtd = 1
    qtds.append(qtd)
    return inds_sorted, np.cumsum(np.array(qtds[:-1])), np.array(qtds[1:]), qtd_unicos

def soma_vetor(vetor):
    return np.sum(vetor)

#########################

class CortaIntervalosQuasiUniforme:

    def __init__(self, vetor, num_div, eh_temporal = False, unit = None):
        
        self.__eh_temporal = eh_temporal
        if(eh_temporal == True):
            valores, qtds, qtd_unicos = unicos_qtds(vetor.view('i8'))
        else:
            valores, qtds, qtd_unicos = unicos_qtds(vetor)
        
        pts_corte, qtds_corte = pontos_corte(qtds, qtd_unicos, num_div)
        pts_corte, self.__qtds_corte = minimiza_desvio_padrao(pts_corte, qtds_corte, qtds, qtd_unicos)
        self.__valores_min, self.__valores_max = calcula_valores_corte(valores, pts_corte)
        self.__valores_medios = calcula_pontos_medios(self.__valores_min, self.__valores_max)
        
        if(eh_temporal):
            self.__valores_min = self.__valores_min.astype('<M8[ns]')
            self.__valores_max = self.__valores_max.astype('<M8[ns]')
            self.__valores_medios = self.__valores_medios.astype('<M8[ns]')
            self.__strings_intervalos = self.__calcula_intervalos_datetime(self.__valores_min, self.__valores_max, unit)
            self.__valores_digitize = self.__valores_max[:-1].view('i8')
        else:
            self.__strings_intervalos = self.__calcula_intervalos_algasignif(self.__valores_min, self.__valores_max)
            self.__valores_digitize = self.__valores_max[:-1]
        
        self.__pares_minimo_maximo = calcula_pares_minmax(self.__valores_min, self.__valores_max)
        
    def __calcula_intervalos_algasignif(self, valores_min, valores_max):
        valores_corte = np.append(valores_min, valores_max[-1])
        alga_signif = 1
        str_conv = ''.join(['%.', str(alga_signif), 'g'])
        cortes_interv = np.array([float(str_conv % valores_corte[i]) for i in range(valores_corte.size)])
        flag = calcula_min_diff(cortes_interv) #não pode ter pontos de corte iguais
        while flag == 0:
            alga_signif += 1
            str_conv = ''.join(['%.', str(alga_signif), 'g'])
            cortes_interv = np.array([float(str_conv % valores_corte[i]) for i in range(valores_corte.size)])
            flag = calcula_min_diff(cortes_interv)
        strings = [str(v) for v in cortes_interv]
        strings = [v if v[-2:] != '.0' else v[:-2] for v in strings]
        return np.array(['['+strings[i]+', '+strings[i+1]+')' for i in range(valores_corte.size-1)])

    def __calcula_intervalos_datetime(self, valores_min, valores_max, unit):
        valores_corte = np.append(valores_min, valores_max[-1])
        strings = [np.datetime_as_string(v, unit = unit) for v in valores_corte]
        return np.array(['['+strings[i]+', '+strings[i+1]+')' for i in range(valores_corte.size-1)])
    
    def pares_minimo_maximo_discretizacao(self):
        return self.__pares_minimo_maximo

    def valores_medios_discretizacao(self):
        return self.__valores_medios

    def strings_intervalos_discretizacao(self):
        return self.__strings_intervalos
    
    def info_discretizacao(self):
        #Retorna um dataframe com as informações relevantes de cada intervalo da discretização
        df = pd.DataFrame(zip(self.__qtds_corte, self.__valores_min, self.__valores_max, self.__valores_medios, self.__strings_intervalos), 
                          columns = ['QTD', 'Min', 'Max', 'Meio', 'Str'])
        return df
    
    def curva_densidade(self):
        valores = np.array([x for y in self.__pares_minimo_maximo for x in y])
        if(self.__eh_temporal):
            fracL = np.repeat(self.__qtds_corte/(self.__valores_max - self.__valores_min).astype(float), 2)
        else:
            fracL = np.repeat(self.__qtds_corte/(self.__valores_max - self.__valores_min), 2)
        return valores, fracL
    
    def curva_distribuicao(self, bins = None):
        if bins is None:
            bins = self.__valores_medios.size
        val_min = self.__valores_min[0]
        val_max = self.__valores_max[-1]
        L = (val_max - val_min)/bins
        valores_corte_bins = [val_min + L*i for i in range(0, bins+1)]
        qtds_bins = [conta_qtds_bin(valores_corte_bins[i], valores_corte_bins[i+1], self.__valores_min, self.__valores_max, self.__qtds_corte) for i in range(0, bins)]
        
        valores_medios_bins = np.array([val_min + L*(i + 1/2) for i in range(1, bins+1)])
        frac_bins = np.array(qtds_bins)/np.sum(self.__qtds_corte)
        return valores_medios_bins, frac_bins, L
   
    def grafico_densidade(self, alpha = 0.5, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, fracL = self.curva_densidade()
            ax.fill_between(valores, fracL, color = paleta_cores[0], alpha = alpha)
            ax.plot(valores, fracL, color = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('Fração/L')
            ax.set_ylim(bottom = 0.0)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()

    def grafico_distribuicao(self, alpha = 0.5, bins = None, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, frac, largura = self.curva_distribuicao(bins = bins)
            ax.bar(valores, frac, color = paleta_cores[0], alpha = alpha, width = largura, linewidth = 2, edgecolor = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('Fração')
            ax.set_ylim(bottom = 0.0)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()
            
    def aplica_discretizacao(self, vetor, usar_ponto_medio = False):
        if(self.__eh_temporal):
            flag_na = np.isnan(vetor)
            disc = discretiza_vetor(vetor.view('i8'), self.__valores_digitize)
            if(usar_ponto_medio):
                disc = disc.astype(int)
                disc = np.array([self.__valores_medios[disc[i]] if ~flag_na[i] else np.datetime64("NaT") for i in range(disc.size)])
                return disc
            else:
                disc[flag_na] = np.nan
                return disc
        else:
            if(usar_ponto_medio):
                return discretiza_vetor_media(vetor, self.__valores_digitize, self.__valores_medios)
            else:
                return discretiza_vetor(vetor, self.__valores_digitize)

########################
 
########################

class CortaIntervalosGanhoInformacao:

    def __init__(self, vetor, alvo, num_div, qtd_min = 0, eh_temporal = False, unit = None, balancear = False):
        
        self.__eh_temporal = eh_temporal
        if(eh_temporal == True):
            valores, qtds, qtds_alvo, qtd_unicos = unicos_qtds_alvos(vetor.view('i8'), alvo)
        else:
            valores, qtds, qtds_alvo, qtd_unicos = unicos_qtds_alvos(vetor, alvo)
        
        pts_corte, qtds_corte, qtds_alvo_corte = pontos_corte_alvo(qtds, qtd_unicos, qtds_alvo, num_div)
        if(balancear):
            pts_corte, self.__qtds_corte, self.__qtds_alvo_corte = minimiza_entropia_balanceada(pts_corte, qtds_corte, qtds, qtds_alvo_corte, qtds_alvo, qtd_unicos, qtd_min)
        else:
            pts_corte, self.__qtds_corte, self.__qtds_alvo_corte = minimiza_entropia(pts_corte, qtds_corte, qtds, qtds_alvo_corte, qtds_alvo, qtd_unicos, qtd_min)
        self.__valores_min, self.__valores_max = calcula_valores_corte(valores, pts_corte)
        self.__valores_medios = calcula_pontos_medios(self.__valores_min, self.__valores_max)
        
        if(eh_temporal):
            self.__valores_min = self.__valores_min.astype('<M8[ns]')
            self.__valores_max = self.__valores_max.astype('<M8[ns]')
            self.__valores_medios = self.__valores_medios.astype('<M8[ns]')
            self.__strings_intervalos = self.__calcula_intervalos_datetime(self.__valores_min, self.__valores_max, unit)
            self.__valores_digitize = self.__valores_max[:-1].view('i8')
        else:
            self.__strings_intervalos = self.__calcula_intervalos_algasignif(self.__valores_min, self.__valores_max)
            self.__valores_digitize = self.__valores_max[:-1]
        
        self.__pares_minimo_maximo = calcula_pares_minmax(self.__valores_min, self.__valores_max)
        
    def __calcula_intervalos_algasignif(self, valores_min, valores_max):
        valores_corte = np.append(valores_min, valores_max[-1])
        alga_signif = 1
        str_conv = ''.join(['%.', str(alga_signif), 'g'])
        cortes_interv = np.array([float(str_conv % valores_corte[i]) for i in range(valores_corte.size)])
        flag = calcula_min_diff(cortes_interv) #não pode ter pontos de corte iguais
        while flag == 0:
            alga_signif += 1
            str_conv = ''.join(['%.', str(alga_signif), 'g'])
            cortes_interv = np.array([float(str_conv % valores_corte[i]) for i in range(valores_corte.size)])
            flag = calcula_min_diff(cortes_interv)
        strings = [str(v) for v in cortes_interv]
        strings = [v if v[-2:] != '.0' else v[:-2] for v in strings]
        return np.array(['['+strings[i]+', '+strings[i+1]+')' for i in range(valores_corte.size-1)])

    def __calcula_intervalos_datetime(self, valores_min, valores_max, unit):
        valores_corte = np.append(valores_min, valores_max[-1])
        strings = [np.datetime_as_string(v, unit = unit) for v in valores_corte]
        return np.array(['['+strings[i]+', '+strings[i+1]+')' for i in range(valores_corte.size-1)])
    
    def pares_minimo_maximo_discretizacao(self):
        return self.__pares_minimo_maximo

    def valores_medios_discretizacao(self):
        return self.__valores_medios

    def strings_intervalos_discretizacao(self):
        return self.__strings_intervalos
    
    def info_discretizacao(self):
        #Retorna um dataframe com as informações relevantes de cada intervalo da discretização
        df = pd.DataFrame(zip(self.__qtds_corte, self.__qtds_alvo_corte, self.__valores_min, self.__valores_max, self.__valores_medios, self.__strings_intervalos), 
                          columns = ['QTD', 'QTD_Alvos', 'Min', 'Max', 'Meio', 'Str'])
        return df
    
    def curva_densidade(self):
        valores = np.array([x for y in self.__pares_minimo_maximo for x in y])
        if(self.__eh_temporal):
            fracL = np.repeat(self.__qtds_corte/(self.__valores_max - self.__valores_min).astype(float), 2)
        else:
            fracL = np.repeat(self.__qtds_corte/(self.__valores_max - self.__valores_min), 2)
        return valores, fracL
    
    def curva_distribuicao(self, bins = None):
        if bins is None:
            bins = self.__valores_medios.size
        val_min = self.__valores_min[0]
        val_max = self.__valores_max[-1]
        L = (val_max - val_min)/bins
        valores_corte_bins = [val_min + L*i for i in range(0, bins+1)]
        qtds_bins = [conta_qtds_bin(valores_corte_bins[i], valores_corte_bins[i+1], self.__valores_min, self.__valores_max, self.__qtds_corte) for i in range(0, bins)]
        
        valores_medios_bins = np.array([val_min + L*(i + 1/2) for i in range(1, bins+1)])
        frac_bins = np.array(qtds_bins)/np.sum(self.__qtds_corte)
        return valores_medios_bins, frac_bins, L
        
    def curva_entropia(self):
        valores = np.array([x for y in self.__pares_minimo_maximo for x in y])
        entropias = np.repeat(entropia_shannon(self.__qtds_alvo_corte/self.__qtds_corte), 2)
        return valores, entropias
   
    def grafico_densidade(self, alpha = 0.5, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, fracL = self.curva_densidade()
            ax.fill_between(valores, fracL, color = paleta_cores[0], alpha = alpha)
            ax.plot(valores, fracL, color = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('Fração/L')
            ax.set_ylim(bottom = 0.0)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()

    def grafico_distribuicao(self, alpha = 0.5, bins = None, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, frac, largura = self.curva_distribuicao(bins = bins)
            ax.bar(valores, frac, color = paleta_cores[0], alpha = alpha, width = largura, linewidth = 2, edgecolor = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('Fração')
            ax.set_ylim(bottom = 0.0)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()
            
    def grafico_entropia(self, alpha = 0.5, rot = None, figsize = [6, 4]):
        paleta_cores = sns.color_palette("colorblind")
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            valores, entropias = self.curva_entropia()
            ax.fill_between(valores, entropias, color = paleta_cores[0], alpha = alpha)
            ax.plot(valores, entropias, color = paleta_cores[0])
            ax.set_xlabel('Valores')
            ax.set_ylabel('Entropia')
            ax.set_ylim(bottom = 0.0)
            #ax.set_ylim(top = 1.01)
            if(rot is not None):
                plt.xticks(rotation = rot)
            plt.show()
            
    def ganho_informacao(self):
        p1 = np.sum(self.__qtds_alvo_corte)/np.sum(self.__qtds_corte)
        entropia_ini = - p1*np.log2(p1) - (1 - p1)*np.log2(1 - p1)
        entropia = np.sum(self.__qtds_corte*entropia_shannon(self.__qtds_alvo_corte/self.__qtds_corte))/np.sum(self.__qtds_corte)
        return 1 - entropia/entropia_ini
        
    def ganho_informacao_balanceado(self):
        entropia = np.sum(entropia_shannon(self.__qtds_alvo_corte/self.__qtds_corte))/self.__qtds_corte.size
        return 1 - entropia
    
    def calcula_ig(self, vetor, alvo):
        p1 = np.sum(alvo)/alvo.size
        entropia_ini = - p1*np.log2(p1) - (1 - p1)*np.log2(1 - p1)
        inds_ordenado, primeira_ocorrencia, qtds_corte, _ = indices_qtds(self.aplica_discretizacao(vetor))
        alvo_agrup = np.split(alvo[inds_ordenado], primeira_ocorrencia[1:])
        qtds_alvo_corte = np.array([soma_vetor(v) for v in alvo_agrup])
        entropia = np.sum(qtds_corte*entropia_shannon(qtds_alvo_corte/qtds_corte))/np.sum(qtds_corte)
        return 1 - entropia/entropia_ini
        
    def calcula_ig_bal(self, vetor, alvo):
        inds_ordenado, primeira_ocorrencia, qtds_corte, _ = indices_qtds(self.aplica_discretizacao(vetor))
        alvo_agrup = np.split(alvo[inds_ordenado], primeira_ocorrencia[1:])
        qtds_alvo_corte = np.array([soma_vetor(v) for v in alvo_agrup])
        entropia = np.sum(entropia_shannon(qtds_alvo_corte/qtds_corte))/qtds_corte.size
        return 1 - entropia
            
    def aplica_discretizacao(self, vetor, usar_ponto_medio = False):
        if(self.__eh_temporal):
            flag_na = np.isnan(vetor)
            disc = discretiza_vetor(vetor.view('i8'), self.__valores_digitize)
            if(usar_ponto_medio):
                disc = disc.astype(int)
                disc = np.array([self.__valores_medios[disc[i]] if ~flag_na[i] else np.datetime64("NaT") for i in range(disc.size)])
                return disc
            else:
                disc[flag_na] = np.nan
                return disc
        else:
            if(usar_ponto_medio):
                return discretiza_vetor_media(vetor, self.__valores_digitize, self.__valores_medios)
            else:
                return discretiza_vetor(vetor, self.__valores_digitize)