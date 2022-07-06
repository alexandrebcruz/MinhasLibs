from flask import Flask
from flask import render_template
from flask import redirect, url_for
import pandas as pd
import math

def df_to_html(df_inp, mask = None):
    if(mask is None):
        df = df_inp
    else:
        df = df_inp[mask]

    styles = [{'selector': 'th', 'props': [('background-color', 'black'), ('color', 'white'),
                                        ('border-style','solid'), ('border-width','0px'),
                                        ('font-size', '12pt')]},
            {'selector': 'tr', 'props': [('background-color', 'white'), ('color', 'black'),
                                        ('font-size', '12pt')]},
            {'selector': '', 'props': [(('width', 'auto'))]}]

    df_styler = df.style.set_table_styles(styles)

    def f_cor(x):
        if x.name == 'Sex':
            def g(i, v):
                if  v == 'male':
                    return ('background-color: red' + '; color: white')
                else:
                    ('background-color: white' + '; color: black')
            return [g(i, v) for i, v in x.iteritems()]
        else:
            return [('background-color: white' + '; color: black') for i, v in x.iteritems()]

    def apply_color(x):
        return df_cores

    df_cores = df.apply(lambda x: f_cor(x), axis = 0)
    df_styler = df_styler.apply(apply_color, axis = None)

    #df_styler = df_styler.hide_index() #Oculta os índices
    df_styler = df_styler.set_properties(**{'text-align': 'center'}) #Centraliza os textos

    html = df_styler.render() #.to_html()
    return html

df_train = pd.read_csv('static/data/train.csv')
df_test = pd.read_csv('static/data/test.csv')

N = 50

app = Flask(__name__)
#Caso queiramos especificar outras rotas padrões para os templates e os arquivos estáticos
#app = Flask(__name__,
#            static_url_path='', 
#            static_folder='/static',
#            template_folder='/templates')

#O router indica qual URL será o gatilho para executar a função (no caso é o URL principal)
@app.route('/')
def index():
    p = 1
    tb = 'train'
    if(session['tabela_atual'] == 'train'):
        tabela_html = df_to_html(df_train.iloc[:N], mask = None)
        session['tot_pags_atual'] = math.ceil(df_train.shape[0]/N)
    elif(session['tabela_atual'] == 'test'):
        tabela_html = df_to_html(df_test.iloc[:N], mask = None)
        session['tot_pags_atual'] = math.ceil(df_test.shape[0]/N)
    return render_template('index.html', tabela = tabela_html, nome_tabela = session['tabela_atual'], pag = session['pag_atual'], tot_pags = session['tot_pags_atual'])

@app.route('/<tabela>/<pag>')
def exibe_tabela(tabela, pag):
    session['pag_atual'] = int(pag)
    session['tabela_atual'] = tabela
    i = session['pag_atual'] - 1
    if(session['tabela_atual'] == 'train'):
        tabela_html = df_to_html(df_train.iloc[i*N:(i+1)*N], mask = None)
        session['tot_pags_atual'] = math.ceil(df_train.shape[0]/N)
    elif(session['tabela_atual'] == 'test'):
        tabela_html = df_to_html(df_test.iloc[i*N:(i+1)*N], mask = None)
        session['tot_pags_atual'] = math.ceil(df_test.shape[0]/N)
    return render_template('index.html', tabela = tabela_html, nome_tabela = session['tabela_atual'], pag = session['pag_atual'], tot_pags = session['tot_pags_atual'])

@app.route('/anterior/<tabela>/<pag>', methods = ['POST'])
def anterior(tabela, pag):
    if(session['pag_atual'] > 1):
        session['pag_atual'] = session['pag_atual'] - 1
    return redirect(url_for('exibe_tabela', tabela = session['tabela_atual'], pag = session['pag_atual']))

@app.route('/proximo/<tabela>/<pag>', methods = ['POST'])
def proximo(tabela, pag):
    pag_atual = int(pag)
    if(session['pag_atual'] < session['tot_pags_atual']):
        session['pag_atual'] = session['pag_atual'] + 1
    return redirect(url_for('exibe_tabela', tabela = session['tabela_atual'], pag = session['pag_atual']))

@app.route('/filtros/', methods = ['POST'])
def filtros():
    if(session['pag_atual'] < session['tot_pags_atual']):
        session['pag_atual'] = session['pag_atual'] + 1
    return redirect(url_for('exibe_tabela', tabela = session['tabela_atual'], pag = session['pag_atual']))
  
#Serve para executar a aplicação sem a necessidade do :
#> set FLASK_APP=application
#> flask run
#Em vez disso, basta executar com o python: python application.py
if __name__ == '__main__':
    app.debug = True #debug = True permite visualizar retornos no console para debug (usar Visual Studio Code)
    app.run(host = '0.0.0.0', port = 2000)
    #O host 0.0.0.0 deixa visível para todos os ips públicos
    #A porta especifica a porta em que a aplicação ficará disponível

#Lembrar que CTRL+C mata a execução no terminal
#pip freeze > requirements.txt #Gera o arquivo de requisitos