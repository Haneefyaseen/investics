import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from flask import Flask, render_template, request, g
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import math

app = Flask(__name__)

scopes = ['https://www.googleapis.com/auth/spreadsheets']
creds = ServiceAccountCredentials.from_json_keyfile_name('secrets.json', scopes)
client = gspread.authorize(creds)

def getShares(range, cell='!B1'):
    shares = client.open_by_key('1SvfBEQJYegxLWJWK85S1QkbLFpvqR4iejiMsigtydQE')
    share = int(shares.values_get(range+cell)['values'][0][0])
    return share

def getSheet(range):
    sheet = client.open_by_key('1SvfBEQJYegxLWJWK85S1QkbLFpvqR4iejiMsigtydQE')
    data = sheet.values_get(range)['values']
    keys = [i[0] for i in data]
    values = [i[1:] for i in data]
    tab =  dict(zip(keys, values))
    return tab

@app.route('/')
def root():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/estimate')
def tool() -> str:
    return render_template('investimator.html')

@app.route('/sheet')
def table():
    return render_template('table.html')

@app.route('/sheet', methods=['GET','POST'])
def sheet():
    range = request.form['range']
    data = getSheet(range)
    del data['LIABILITIES']
    del data['SHARES']
    df = pd.DataFrame.from_dict(data, orient='index')
    return render_template('table.html', sdata = df.to_html(classes='table table-bordered'))

@app.route('/forecast', methods=['GET','POST'])
def forecast():
    num = request.form['t-range']
    range = request.form['stock']
    data = getSheet(range)
    df = pd.DataFrame(data, index=data['YEAR'])
    df = df.replace(',','',regex=True)
    # revenue = Holt(pd.DataFrame([i.replace(',', '') for i in data['REVENUE']]).tail(5).astype(float)).fit().forecast(3).round(2)
    revenue = Holt(df['REVENUE'].tail(5).astype(float)).fit().forecast(int(num)).round(2)
    profit = Holt(df['NET PROFIT'].tail(5).astype(float)).fit().forecast(int(num)).round(2)
    assets = Holt(df['TOTAL ASSETS'].tail(5).astype(float)).fit().forecast(int(num)).round(2)
    lib = Holt(df['LIABILITIES'].tail(5).astype(float)).fit().forecast(int(num)).round(2)
    s_equity = assets - lib
    # pr = ARIMA(df['PRICE-TO-REVENUE'].tail(5).astype(float), order=(0,0,3)).fit().forecast(int(num)).round(2)
    # pe = ARIMA(df['PRICE-TO-EARNINGS'].tail(5).astype(float), order=(0,0,3)).fit().forecast(int(num)).round(2)
    # pb = ARIMA(df['PRICE-TO-BOOK'].tail(5).astype(float), order=(0,0,3)).fit().forecast(int(num)).round(2)
    pr = df['PRICE-TO-REVENUE'].tail(5).astype(float).mean().round(2)
    pe = df['PRICE-TO-EARNINGS'].tail(5).astype(float).mean().round(2)
    pb = df['PRICE-TO-BOOK'].tail(5).astype(float).mean().round(2)
    pred = pd.concat([revenue, profit, assets, s_equity], axis=1, keys=[str.upper(i) for i in ['revenue', 
    'profit', 'assets', 'shareholders-equity']])
    p_data = pd.DataFrame([pr,pe,pb], index=['PRICE-TO-REVENUE', 'PRICE-TO-EARNINGS', 'PRICE-TO-BOOK'], columns=[''])
    print(p_data)
    del data['LIABILITIES']
    del data['SHARES']
    dff = pd.DataFrame.from_dict(data, orient='index')
    year = pd.to_datetime(pred.index).strftime('%Y')
    pred.index = np.asarray(year)
    return render_template('forecast.html', sdata=dff.to_html(classes='table table-bordered'),
    pred=pred.transpose().to_html(classes='table table-bordered'), price_data = p_data.transpose().to_html(classes='table table-bordered'))

@app.route('/sp', methods=['GET','POST'])
def stock():
    num = request.form['t-range']
    range = request.form['stock']
    data = getSheet(range)
    share = getShares(range)
    df = pd.DataFrame(data, index=data['YEAR'])
    df = df.replace(',','',regex=True)
    # revenue = Holt(pd.DataFrame([i.replace(',', '') for i in data['REVENUE']]).tail(5).astype(float)).fit().forecast(3).round(2)
    revenue = Holt(df['REVENUE'].tail(5).astype(float)).fit().forecast(int(num)).round(2)
    profit = Holt(df['NET PROFIT'].tail(5).astype(float)).fit().forecast(int(num)).round(2)
    assets = Holt(df['TOTAL ASSETS'].tail(5).astype(float)).fit().forecast(int(num)).round(2)
    lib = Holt(df['LIABILITIES'].tail(5).astype(float)).fit().forecast(int(num)).round(2)
    s_equity = assets - lib
    pr = df['PRICE-TO-REVENUE'].tail(5).astype(float).mean().round(2)
    pe = df['PRICE-TO-EARNINGS'].tail(5).astype(float).mean().round(2)
    pb = df['PRICE-TO-BOOK'].tail(5).astype(float).mean().round(2)
    price_pr = ((revenue*math.pow(10,7)*pr)/share).round(2)
    price_pe = ((profit*math.pow(10,7)*pe)/share).round(2)
    price_pb = ((s_equity*math.pow(10,7)*pb)/share).round(2)
    avg_price = ((price_pr+price_pb+price_pe)/3).round(2)
    pred = pd.concat([revenue, profit, assets, s_equity], axis=1, keys=[str.upper(i) for i in ['revenue', 
    'profit', 'assets', 'shareholders-equity']])
    stock = pd.concat([price_pr, price_pe, price_pb, avg_price], axis=1, keys=[str.upper(i) for i in ['price based on pr',
    'price based on pe', 'price based on pb', 'average price']])
    p_data = pd.DataFrame([pr,pe,pb], index=['PRICE-TO-REVENUE', 'PRICE-TO-EARNINGS', 'PRICE-TO-BOOK'], columns=[''])
    del data['LIABILITIES']
    del data['SHARES']
    dff = pd.DataFrame.from_dict(data, orient='index')
    year = pd.to_datetime(pred.index).strftime('%Y')
    pred.index = np.asarray(year)
    stock.index = np.asarray(year)
    return render_template('stockprice.html', sdata=dff.to_html(classes='table table-bordered'),pred=pred.transpose().to_html(classes='table table-bordered'),
    stock=stock.transpose().to_html(classes='table table-bordered'), price_data = p_data.transpose().to_html(classes='table table-bordered'))