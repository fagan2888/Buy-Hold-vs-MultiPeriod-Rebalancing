import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('qt5agg') #macosx/ qt5agg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtest import Strategy,Portfolio

class RebalanceVSBuyHold(Strategy):

    def __init__(self,df,names,rbtype):
        self.df = df
        self.names = names
        self.rbtype = rbtype

    def generate_signals(self):
        self.df = self.df.resample(self.rbtype).last()
        sig = pd.DataFrame(index=self.df.index)

        for index,x in enumerate(self.names):
            sig[x] = 1

        return sig



class MarketOnClosePortfolio(Portfolio):

    def __init__(self,sig,df,names,rbtype):
        self.sig = sig
        self.df = df
        self.names = names
        self.rbtype = rbtype
        self.positions = self.generate_positions()

    def generate_positions(self):
        pctchg = self.df.resample(self.rbtype).last().pct_change().dropna()
        self.sig = (self.sig[1:] * (1/len(self.names))).round(2)
        positions = pctchg.mul(self.sig.values)
        return positions


    def backtest_portfolio(self):
        rtns = self.positions.sum(axis=1)
        return rtns


if __name__ == '__main__':


    df = pd.read_csv('all_asset_class.csv', index_col='Date', parse_dates=True)
    df.drop(['dollar', 'yc', 'senti', '7yTR', '10yTR', '30yTR'], axis=1, inplace=True)
    df.ffill(inplace=True)
    df.columns = ['Crude', 'Gold', 'DM Equity', 'EM Corp', 'EM Equity', 'TSY', '$Corp', '$HY', '$BBB']

    stocks = ['Crude', 'Gold', 'DM Equity', 'EM Corp', 'EM Equity', 'TSY', '$Corp', '$HY', '$BBB']
    df1 = df['2003'].tail(1)
    df = df['2004':'2018']
    df = df1.append(df)

    #rebalancing type
    rb_type = ['BM','BQ','BA']
    rtn = []
    for rt in rb_type:
        rbBH = RebalanceVSBuyHold(df,stocks,rt)
        sig = rbBH.generate_signals()
        port = MarketOnClosePortfolio(sig,df,stocks,rt)
        returns = port.backtest_portfolio()
        returns_cum_ = np.cumproduct(1+returns)-1
        rtn.append(returns_cum_)
        returns_cum_.plot(label=rt, legend=True)


    #buy and hold graph

    dff = df.resample('BQ').last()
    dff = dff.pct_change()
    dff = dff.replace(np.nan, 0)
    dff1 = pd.DataFrame()
    for x in dff.columns:
        dff1[str(x) + str('d')] = dff[x] + 1

    dff2 = pd.DataFrame()
    for x in dff1.columns:
        dff2[str(x) + str('d')] = np.cumprod(dff1[x])

    dff2['sum'] = np.sum(dff2, axis=1)
    dff2['diff'] = (dff2['sum'] / dff2['sum'].shift(1)) - 1
    dff2.dropna(inplace=True)
    returns_cum_ = np.cumproduct(dff2['diff'] + 1) - 1

    returns_cum_.plot(label='BH', legend=True)


    #total returns

    print('Total returns for the portfolio with monthly rebalancing is {:.2f}%'.format(np.asscalar(rtn[0].tail(1))*100))
    print('Total returns for the portfolio with quarterly rebalancing is {:.2f}%'.format(np.asscalar(rtn[1].tail(1))*100))
    print('Total returns for the portfolio with annual rebalancing is {:.2f}%'.format(np.asscalar(rtn[2].tail(1))*100))
    print('Total returns for the portfolio with buyandhold is {:.2f}%'.format(np.asscalar(returns_cum_.tail(1)) * 100))

    print('\n')
    tenor = ['Monthly', 'Quarterly', 'Annual','BuyandHold']
    returns = [np.asscalar(rtn[0].tail(1)) * 100, np.asscalar(rtn[1].tail(1)) * 100, np.asscalar(rtn[2].tail(1)) * 100,
               np.asscalar(returns_cum_.tail(1)) * 100]

    table = pd.DataFrame(dict(Tenor=tenor, Returns=returns))
    table.set_index('Tenor',inplace=True)
    table.sort_values(by='Returns', inplace=True)
    table.plot(kind='barh')
    print(table)
    plt.show()