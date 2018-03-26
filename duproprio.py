#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Usage:
  duproprio  [options]
  duproprio dev  [options]

Options:
  -h --help      Show this screen.
  -f --force-update    Overwrite/recalculate database tables and outputs
  -i --index-path      Folder containing index pages
  -d --details-path      Folder containing details pages

This just uses a manual search of recent solds, or other list, to get locations for sold properties

"""

import glob,os,re
import numpy as np
import urllib2
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from cpblUtilities.pandas_utils import str2df
from cpblUtilities import doSystemLatex

def LaTeX_draw_rooms(xys=None):
    if xys is None:
        xys = str2df("""Room	width	height
Entrance	0.97	4.95
Balcony	1.52	1.22
Bathroom	3.18	2.13
Bedroom 1 (Master)	3.05	3.66
Bedroom 2	3.48	3.15
Hall	3.78	2.08
Kitchen	3.3	3.61
Laundry room	0.91	1.12
Living / Dining room	7.16	3.73
Office	3.48	1.93
Storage space	1.22	1.35
Terrace	3.05	2.44
Walk In	1.68	0.74
Walk In	2.26	0.61
Walk-In Closet	2.13	1.35""")
    texout = r"""
\documentclass{article}
\usepackage{tikz}
\begin{document}

    """ + '\n'.join([r"""  \tikz \draw (0,0) rectangle ({x},{y})  node[pos=.5] {{{text}}};  ~ ~  """.format(x= arow['width'],y= arow['height'], text= arow['Room']) for ii,arow in xys.iterrows()
        ])+ r"""
\end{document}

    """
    
    doSystemLatex('tmp', texout,
                  launchDaemon=False, # keep watching for the source file to change
                  display = True,)
    
    wowo



def use_index_pages_to_get_sold_property_pages(named_manual_search='sold-manual'):
    urls,             salefeatures =[],[]
    for file in glob.glob(named_manual_search+'/'+"*.html"):
        html = open(file,'rt').read()
        urls += re.findall("""<a href="(.*?)" class="gtm-search-results-link-property search-results-listings-list__item-image-link" property="significantLink">""",html)
        # Sold dates only available in this index page?
        items = re.findall("""<li id="listing-([0123456789]*)" class="search-results-listings-list__item(.*?)search-results-listings-list__item-footer""", html, re.DOTALL)
        
        for uid,rest in items:
            lat = re.findall('"latitude" content="([-.0123456789]*)"', rest)[0]
            lon = re.findall('"longitude" content="([-.0123456789]*)"', rest)[0]
            url =  re.findall("""<a href="(.*?)" class="gtm-search-results-link-property search-results-listings-list__item-image-link" property="significantLink">""",rest)[0]
            timesold,datesold = re.findall('Sold in : <strong>(.*?) on (.*?)</strong></div>', rest)[0]

            salefeatures += [dict(latitude= float(lat),
                                  longitude = float(lon),
                                  url = url,
                                  timesold = timesold,
                                  datesold=datesold,
                                  uid = uid,)]
    dfsf = pd.DataFrame(salefeatures).set_index('uid', drop=False)

    ud =named_manual_search+'/units/'
    os.system('mkdir -p '+ ud)
    #os.chdir('units')
    for url in urls:
        pname = url.split('-')[-1]
        ds = dfsf.loc[pname]
        fname =ud+pname+'.html'
        if not os.path.exists(fname):
            response = urllib2.urlopen(url)
            html = response.read()
            with open(fname,'wt') as fout:
                fout.write(html)
        else:
            html=  open(fname,'rt').read()

        try: 
            mtime = os.path.getmtime(fname)
        except OSError:
            mtime = 0

        ds['download_date'] = datetime.fromtimestamp(mtime)
        
        yield ds,html


def extract_data_from_sold_property_pages():
    features=[]
    roomfeatures=[]
    for prec,html in use_index_pages_to_get_sold_property_pages():
        # Redundant:
        #latlon = re.findall("""{"latitude":(.*?),"longitude":(.*?),""", html)
        roomdirs = re.findall("""listing-rooms-details__table__item--room">\n *([^\n]*).*?listing-rooms-details__table__item--dimensions__content"> *\n *([^\n]*)""",html, re.DOTALL)

        lchars = re.findall("""listing-list-characteristics__row listing-list-characteristics__row--label">(.*?)</div>.*?listing-list-characteristics__row listing-list-characteristics__row--value">(.*?)</div>""", html, re.DOTALL)

        # These come in different flavours; so use two steps
        mainchars_ = re.findall("""<div class="listing-main-characteristics__(?:label|item-dimensions)">(.*?)</div>""", html, re.DOTALL)
        mainchars = []
        for mm in mainchars_:
            title = re.findall("""<span class="listing-main-characteristics__title.*?>(.*?)</span>""", mm, re.DOTALL)[0].strip()
            value = re.findall("""<span class="listing-main-characteristics__number.*?>(.*?)</span>""", mm, re.DOTALL)[0].strip()
            mainchars+= [[title,value] ]

        listprice = re.findall("""<div class="listing-price">.*?\$([^\n]*)""", html, re.DOTALL)[0].replace(',','')
        
        features += [prec.combine_first(  pd.Series(dict(lchars+mainchars+[['listprice',listprice]])) ) ]

        for rd in roomdirs:
            roomfeatures += [dict([
                ['room', rd[0]],
                ['dim', rd[1]],
                ['uid', prec['uid']],
            ])]

    return features, roomfeatures

def area_from_duproprio_dimensions(aline):
    if pd.isnull(aline):# in [np.nan]:
        return np.nan
    if 'm)' in aline:
        mm= re.findall('([,.0123456789]*) m x ([,.0123456789]*) m' , aline)
        x,y =  float(mm[0][0].replace(',','.')) ,  float( mm[0][1].replace(',','.'))
        return x,y,x*y # m, m, metres squared
    if 'm²' in aline:
        return float(re.findall('([0123456789.]*) m²', aline)[0])  # metres squared
    ohoh


def prep_data(forceUpdate=False):
    storedata = 'tmpsoldd.pandas'
    if os.path.exists(storedata) and not forceUpdate:
        return pd.read_pickle(storedata), pd.read_pickle('all'+storedata)
    if 1: #not os.path.exists('soldprops.pandas') or forceUpdate:
        ff,rf =extract_data_from_sold_property_pages()
        pd.concat([pd.DataFrame(aff).T for aff in ff]).to_pickle('soldprops.pandas')
        pd.DataFrame(rf).to_pickle('soldpropsrooms.pandas')

    df= pd.read_pickle('soldprops.pandas').set_index('uid')
    rdf= pd.read_pickle('soldpropsrooms.pandas')

    rdf['area']= rdf.dim.map(lambda ss: area_from_duproprio_dimensions(ss)[2])

    rdf['outdoor'] = rdf.room.apply(lambda ss: ss.lower() in ['shed','driveway','terrace','deck','patio','balcony','garage'])
    rdf['indoor'] = rdf.room.apply(lambda ss: ss.lower() in 
    ['bathroom', 'bedroom 1 (master)', 'bedroom 2',                        
           'dining room', 'kitchen', 'laundry room', 'living room',                        
           'mezzanine', 'bedroom 3', 'utility',                                 
           'dining room / living room', 'bedroom',                     
            'entrance', 'kitchenette', 'walk-in closet',                        
           'open concept', 'walk in', 'workshop', 'family room',                
           'hall', 'two piece bathroom', 'storage space',                                  
           'dining room / kitchen', 'storage', 'office',                         
           'living / dining room', 'lounge', 'suite', 'ensuite',                           
           'recreation room', 'sitting room', 'suite entrance', 
           'nook', 'bedroom 4', 'patio', 'meter room', 'apartment 1',                      
           'eat-in kitchen', 'apartment bedroom', 'basement living room',                  
           'wine cellar'],)
    print('Following are not classified:')
    # 'Basement','Other',                   
    print rdf.query('indoor==False').query('outdoor==False')[['room','area']]
    df['indoorArea']= rdf.query('indoor==True').groupby('uid').area.sum()
    df['outdoorArea']= rdf.query('outdoor==True').groupby('uid').area.sum()
    df.outdoorArea.fillna(0, inplace=True)
    df.listprice = df.listprice.astype(int)

    from cpblUtilities.mapping import Position
    mylat=45.51892
    mylon=-73.58966
    fp = Position(mylon, mylat)
    df['distance'] = df.apply(lambda adf: (Position(adf.longitude, adf.latitude)-fp)*1000,
                              axis=1)

    # Extract total area:
    df['livingSpace'] = df['Living space area (basement exclu.)'].map( area_from_duproprio_dimensions)

    # Fix Divided/undivided
    df['divided'] = df.Ownership.fillna(
        df['Property Style'].map(lambda ss: {'Undivided Co-Ownership': 'Undivided',
                                                     'Divided Co-Ownership': 'Divided',}.get(ss,'unk')))
    df['undivided'] = (df.divided=='Undivided').astype(int)

    # Bathrooms, etc:
    df['bathrooms'] = df['bathrooms'].fillna(df.bathroom).astype(float)
    df['bedrooms'] = df['bedrooms'].fillna(df.bedroom).astype(float)
    df['levels'] = df['levels'].fillna(df.level).astype(float)
    df['parkingExt'] = df['Number of exterior parking'].fillna(0).astype(int)
    df['parkingInt'] = df['Number of interior parking'].fillna(0).astype(int)
    df['parkings'] = df['parkingExt']+ df['parkingInt'] 
    df['firstfloor'] = df['Located on which floor? (if condo)']=='1'
    df['secondfloor'] = df['Located on which floor? (if condo)']=='2'
    df['thirdfloor'] = df['Located on which floor? (if condo)']=='3'

    
    # Key variables
    dfk = df[['distance', 'indoorArea','outdoorArea', 'divided', 'livingSpace', 'listprice', 'bedrooms','bathrooms','levels', 'parkings','parkingInt', 'parkingExt',
              'firstfloor','secondfloor', 'thirdfloor',
              'undivided', 'latitude', 'longitude',
    ]].sort_values('distance')
    dfk['ppsqft'] = dfk.listprice/(dfk.livingSpace* 3.28084*3.28084)

    dfk.to_pickle(storedata)
    df.to_pickle('all'+storedata)
    return dfk, df


#########################################################################################################
#=======================================================================================================#
#===================================  MAIN =============================================================#
#=======================================================================================================#
#=======================================================================================================#
#########################################################################################################
defaults={}
if __name__ == '__main__':
    # Docopt is a library for parsing command line arguments
    try:
        # Parse arguments, use file docstring as a parameter definition
        arguments = docopt.docopt(__doc__)
        knownmodes = [aa for aa in arguments if not aa.startswith('-')]
        #if arguments['--serial']:
        #        defaults['server']['parallel']=False
        defaults['forceUpdate'] = arguments['--force-update'] == True
        runmode=''.join([ss*arguments[ss] for ss in knownmodes])
        runmode= None if not runmode else runmode
    # Handle invalid options
    except docopt.DocoptExit as e:
        print e.message



    if runmode is None: 
        dfk,df = prep_data(False)

        dfku = dfk[dfk.divided=="Undivided"]
        dfkd = dfk[dfk.divided=="Divided"]


        dfk.to_stata('tmp.dta')
        # Analysis

        # Difference in prices based only on area?
        fig,ax = plt.subplots(1)
        for dd in ['Divided','Undivided']:
            yy = dfk.query('divided=="{}"'.format(dd))
            plt.semilogx(yy.listprice, yy.indoorArea, '.', label=dd)
        plt.legend()
        plt.savefig('price-div.pdf')

        # sum of area matchs living space?
        ax = dfk[['indoorArea','livingSpace']].plot.scatter('indoorArea','livingSpace')
        ax.plot(ax.get_xlim(), ax.get_xlim(), 'k', zorder =-10)
        ax.set_xlabel('Indoor area')
        ax.set_ylabel('Living space')
        plt.savefig('area-agreement.pdf')


        fig,ax = plt.subplots(1)
        #plt.style.use('seaborn-deep')
        bins = np.linspace(200, 800, 31)
        ax.hist([dfku.ppsqft, dfkd.ppsqft], bins, label=['Undivided', 'Divided'])
        ax.legend(loc='upper right')
        ax.set_xlabel('Price/square-foot')
        plt.savefig('ppsqft-div.pdf')

        plt.show()


        #dfk.groupby('divided')['ppsqft'].hist(ax=ax)
        #ax.legend()





        dodoo
        foiu
    elif runmode in ['dev']:
        
