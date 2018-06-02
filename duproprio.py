#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Usage:
  duproprio  [options]
  duproprio dev  [options]
  duproprio indexes [options]
  duproprio details [options]
  duproprio assess [option]
  duproprio prep [options]
  duproprio plot [options]
  duproprio stata [options]
  duproprio photos [options]
  duproprio convert_areas [options]

Options:
  -h --help      Show this screen.
  -f --force-update    Overwrite/recalculate database tables and outputs
  -i --index-path=<indexpath>      Folder containing index pages
  -d --details-path=<detailspath>      Folder containing details pages

duproprio indexes : Start from index files in a given folder
duproprio details :  Start from details files in a given folder
duproprio assess  : Assess (estimate) value of a given URL or duproprio ID.
duproprio prep    : Get all data; concatenate it...

duproprio convert_areas: A completely unrelated mode; this accepts pasted room areas from a Duproprio page, and adds up the area to find the total

This just uses a manual search of recent solds, or other list, to get locations for sold properties.
That is, there is no scraping involved.
However, if you create your own search and download the index files corresponding to your search, this helps to analyse the data from the listed units.


Examples:

Download individual property files, from some index pages from a custom search
 run duproprio.py details --details-path=current-manual

Download a lot of photos:
 run duproprio.py photos --details-path=current-manual/units



to do:
 - get latex stata working.
 - then evaluate units.

"""
import docopt
import glob,os,re
import numpy as np
import urllib2
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from cpblUtilities.pandas_utils import str2df
from cpblUtilities import defaults, doSystemLatex
paths=defaults['paths']
paths['batch'] ='./'
import pystata as pst

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



def use_index_pages_to_get_sold_property_pages(indexpath=None):
    if indexpath is None:
        indexpath = paths['indexhtml'] #named_manual_search='sold-manual'):
    urls,             salefeatures =[],[]
    for file in glob.glob(indexpath+'/'+"*.html"):
        html = open(file,'rt').read()
        urls += re.findall("""<a href="(.*?)" class="gtm-search-results-link-property search-results-listings-list__item-image-link" property="significantLink">""",html)
        # Sold dates only available in this index page?
        items = re.findall("""<li id="listing-([0123456789]*)" class="search-results-listings-list__item(.*?)search-results-listings-list__item-footer""", html, re.DOTALL)
        
        for uid,rest in items:
            lat = re.findall('"latitude" content="([-.0123456789]*)"', rest)[0]
            lon = re.findall('"longitude" content="([-.0123456789]*)"', rest)[0]
            url =  re.findall("""<a href="(.*?)" class="gtm-search-results-link-property search-results-listings-list__item-image-link" property="significantLink">""",rest)[0]
            soldinfo = re.findall('Sold in : <strong>(.*?) on (.*?)</strong></div>', rest)
            timesold,datesold = '','' if not soldinfo else soldinfo[0]

            salefeatures += [dict(latitude= float(lat),
                                  longitude = float(lon),
                                  url = url,
                                  timesold = timesold,
                                  datesold=datesold,
                                  uid = uid,)]
    dfsf = pd.DataFrame(salefeatures).set_index('uid', drop=False)

    ud = indexpath+'/units/'
    #assert paths.get('detailshtml',None) in [ None, paths['indexhtml']+'/units/']
    #paths['detailshtml'] = paths['indexhtml']+'/units/'
    #ud = paths['detailshtml']
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


def extract_data_from_index_pages(indexpath= None, forceUpdate=False):
    """
    Return list of features and list of rooms from all properties listed in all index pages in a given folder.
    """
    if indexpath is None:
        indexpath = paths['indexhtml'] #named_manual_search='sold-manual'):
    storedata = indexpath+'all_records.pandas'
    storedatarooms = indexpath+'all_records_rooms.pandas'
    if os.path.exists(storedata) and not forceUpdate and not defaults['forceUpdate']:
        return pd.read_pickle(storedata), pd.read_pickle(storedatarooms)
    
    features=[]
    roomfeatures=[]
    for prec,html in use_index_pages_to_get_sold_property_pages(indexpath):
        feat, rfeat = extract_data_from_duproprio_detailed_property_html(html)
        #prec = prec.drop_duplicates() # WHAT! Why is this necessary for some weird cases
        if isinstance(prec, pd.DataFrame):
            print('wlwlejrlwklew WHAT HAS HAPPEND?? prec has multiple values for ')
            print(prec)
            continue
        print(len(prec))
        rf = pd.DataFrame(rfeat)
        rf['uid']= prec['uid']
        roomfeatures+=[rf]
        
        features += [prec.combine_first( feat )]

    rff =   pd.concat(roomfeatures)
    rff.to_pickle(storedatarooms)
    bff =   pd.DataFrame(features)
    bff.to_pickle(storedata)
    return bff, rff

    sowowo # maybe rewrite this concat to just dataframe a list of series.
    #then save it.
    ###ff,rf =extract_data_from_index_pages()
    pd.concat([pd.DataFrame(aff).T for aff in ff]).to_pickle('soldprops.pandas')
    pd.DataFrame(rf).to_pickle('soldpropsrooms.pandas')
        
    return features, pd.concat(roomfeatures)


    storedata = 'tmpsoldd.pandas'



def extract_data_from_duproprio_detailed_property_html(html):

    costs_ = re.findall(r"""<div class="mortgage-data__table__row__item mortgage-data__table__row__item--name">\s*([^\n]*)\s*</div>\s*<div class="mortgage-data__table__row__item mortgage-data__table__row__item--monthly-costs">\s*([^\n]*)\s*</div>\s*<div class="mortgage-data__table__row__item mortgage-data__table__row__item--yearly-costs">\s*([^\n]*)\s*</div>""", html, re.DOTALL)
    # For the moment, let's assume everything is available in annual form:
    costs = [[cc[0].replace(' ',''), float(cc[2].replace('$','').replace(',',''))] for cc in costs_]
    if costs: # Don't add sum where there are none found
        costs+= [['totalAnnualCosts', sum([cc[1] for cc in costs])]]

    roomdirs = re.findall("""listing-rooms-details__table__item--room">\n *([^\n]*).*?listing-rooms-details__table__item--dimensions__content"> *\n *([^\n]*)""",html, re.DOTALL)

    lchars = re.findall("""listing-list-characteristics__row listing-list-characteristics__row--label">(.*?)</div>.*?listing-list-characteristics__row listing-list-characteristics__row--value">(.*?)</div>""", html, re.DOTALL)

    # These come in different flavours; so use two steps
    mainchars_ = re.findall("""<div class="listing-main-characteristics__(?:label|item-dimensions)">(.*?)</div>""", html, re.DOTALL)
    mainchars = []
    for mm in mainchars_:
        title = re.findall("""<span class="listing-main-characteristics__title.*?>(.*?)</span>""", mm, re.DOTALL)[0].strip()
        value = re.findall("""<span class="listing-main-characteristics__number.*?>(.*?)</span>""", mm, re.DOTALL)[0].strip()
        mainchars+= [[title,value] ]

    # Redundant?
    latlon = re.findall("""{"latitude":(.*?),"longitude":(.*?),""", html)
    if latlon:
        mainchars+= [['latitude',float(latlon[0][0])], ['longitude',float(latlon[0][1])]]
    check_for_listprice = re.findall("""<div class="listing-price">.*?</div>""", html, re.DOTALL)
    listprice = re.findall("""<div class="listing-price">.*?\$([^\n]*)""",
                           check_for_listprice[0], re.DOTALL)
    listprice = np.nan if not listprice else float(listprice[0].replace(',',''))

    features1 =  pd.Series(dict(lchars+mainchars+costs+[['listprice',listprice]]) ) 

    roomfeatures1=[]
    for rd in roomdirs:
        roomfeatures1 += [dict([
            ['room', rd[0]],
            ['dim', rd[1]],
            #['uid', prec['uid']],
        ])]

    return features1, roomfeatures1


def area_from_duproprio_dimensions(aline):
    if pd.isnull(aline):# in [np.nan]:
        return np.nan
    if 'm)' in aline:
        mm= re.findall('([,.0123456789]*) m x ([,.0123456789]*) m' , aline)
        x,y =  float(mm[0][0].replace(',','.')) ,  float( mm[0][1].replace(',','.'))
        return x,y,x*y # m, m, metres squared
    if 'm²' in aline:
        return float(re.findall('([0123456789.]*) m²', aline)[0])  # metres squared
    raise(Error('Cannot recognize dimensions format'))

def get_photos_for_folder_of_details_pages(detailspath=None):
    if detailspath is None:
        detailspath = paths['detailshtml']
    features, roomfeatures =[],[]
    for filen in glob.glob(detailspath+'/'+"*.html"):
        html = open(filen,'rt').read()
        imagepath = filen.replace('.html','_photos/')
        os.system('mkdir -p '+ imagepath)
        for photo in re.findall("""{"is_primary"(.*?)}""",html):
            typedesc = re.findall('"type":"(.*?)","description":"(.*?)"', photo)
            formats = re.findall('[{,]"(.*?)":"(.*?.jpg)', photo.split('formats":')[1])
            if formats[-1][1] in ['\\/build\\/img\\/jpg']:
                continue
            if '/' in formats[-1][1]:
                photofile = formats[-1][1].split('/')[-1]
                url = 'https://photos.duproprio.com/'+ formats[-1][1].replace(r'\/','/')
            else:
                photofile=formats[-1][1]
                url = 'https://photos.duproprio.com/'+ photofile
            if  os.path.exists(imagepath+photofile):
                pass#print(imagepath+' Already have '+url)
            else:
                try:
                    response = urllib2.urlopen(url)
                    jpg = response.read()
                    with open(imagepath+photofile,'wb') as fout:
                        fout.write(jpg)
                    print(photofile)
                except urllib2.HTTPError as e:
                    print(imagepath+' FAILED to get '+url)


def process_folder_of_details_pages(detailspath=None):
    if detailspath is None:
        detailspath = paths['detailshtml']
    features, roomfeatures =[],[]
    for filen in glob.glob(detailspath+'/'+"*.html"):
        html = open(filen,'rt').read()
        url = re.findall("""<meta property="og:url" content="(.*?)">""",html)[0]
        uid = url.split('-')[-1]
        settargeting_etc = re.findall(""".setTargeting.'(.*?)','(.*?)'""", html) +    re.findall("""<meta property="(.*?)" content="(.*?)".""", html) 
        
        feat, rfeat = extract_data_from_duproprio_detailed_property_html(html)

        rf = pd.DataFrame(rfeat)
        rf['uid']= uid
        roomfeatures+=[pd.DataFrame(rf)]
        
        feat['uid'] =  uid
        features += [ feat.combine_first(pd.Series(dict(settargeting_etc))) ]

    return pd.DataFrame(features).set_index('uid',drop=False) , pd.concat(roomfeatures)

def process_features_to_final_dataframe(features, roomfeatures):
    """
    Take two data frames; massage things to generate a small summary dataset
    """
    rdf = roomfeatures
    df = features
    
    rdf['area']= rdf.dim.map(lambda ss: area_from_duproprio_dimensions(ss)[2])

    rdf['outdoor'] = rdf.room.apply(lambda ss: ss.lower() in ['shed','driveway','terrace','deck','patio','balcony','garage'])
    rdf['indoor'] = rdf.room.apply(lambda ss: ss.lower() in 
    ['bonus room','great room', 'den', 'tv room','apartment living room', 'tv room', 'office front', 'bathroom', 'bedroom 1 (master)', 'bedroom 2',                        
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

    """Basement',  'Great room', 'Basement', , 'Other',  '"""

    print('Following are not classified:')
    # 'Basement','Other',                   
    print rdf.query('indoor==False').query('outdoor==False')[['room','area']]
    df['indoorArea']= rdf.query('indoor==True').groupby('uid').area.sum()
    df['outdoorArea']= rdf.query('outdoor==True').groupby('uid').area.sum()
    df.outdoorArea.fillna(0, inplace=True)
    df.listprice = df.listprice.astype(float)

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
    if 'bedroom' not in df: df['bedroom'] = np.nan
    df['bedrooms'] = np.nan if 'bedrooms' not in df else df['bedrooms'].fillna(df.bedroom).astype(int)

    df['levels'] = df['levels'].fillna(df.level).astype(float)
    if 'Number of exterior parking' in df:
        df['parkingExt'] = df['Number of exterior parking'].fillna(0).astype(int)
    else:
        df['parkingExt'] = 0
    if 'Number of interior parking' in df:
        df['parkingInt'] = df['Number of interior parking'].fillna(0).astype(int)
    else:
        df['parkingInt'] = 0
    df['parkings'] = df['parkingExt']+ df['parkingInt'] 
    df['firstfloor'] = df['Located on which floor? (if condo)']=='1'
    df['secondfloor'] = df['Located on which floor? (if condo)']=='2'
    df['thirdfloor'] = df['Located on which floor? (if condo)']=='3'
    if 'streetAddress' in df:
        df['streetAddress']=     df['streetAddress'].str.replace('&#39;',"'")

    # Key variables
    dfk = df[['distance', 'indoorArea','outdoorArea', 'divided', 'livingSpace', 'listprice', 'bedrooms','bathrooms','levels', 'parkings','parkingInt', 'parkingExt',
              'firstfloor','secondfloor', 'thirdfloor',
              'undivided', 'latitude', 'longitude', ]+('totalAnnualCosts' in df)*[
              'Propertytaxes', 'Schooltaxes', 'Electricity', 'Condofees', 'totalAnnualCosts', ]].sort_values('distance')
    dfk['latitude'] = dfk['latitude'].astype(float)
    dfk['longitude'] = dfk['longitude'].astype(float)
    dfk['ppsqft'] = dfk.listprice/(dfk.livingSpace* 3.28084*3.28084)
    dfk['ppsqft_r'] = dfk.listprice/(dfk.indoorArea* 3.28084*3.28084)
    return dfk, df
    
def add_stata_modeled_to_df(df): # df has index "uid"
    if os.path.exists(pst.WPdta('allunitsModeled')):
        dfm=pst.dta2df(pst.WPdta('allunitsModeled')).set_index('uid')
        dfm = dfm[~dfm.index.duplicated(keep='first')]
        assert len(dfm.index) == len(dfm.index.unique())
        assert len(df.index) == len(df.index.unique())
        foo= df.join( dfm[['selistpricehat', 'listpricehat']])
        assert len(foo.index)== len(foo.index.unique())
        assert len(df.join( dfm[['selistpricehat', 'listpricehat']])) == len(df.join( dfm[['selistpricehat', 'listpricehat']]).drop_duplicates())
        
    return df.join( dfm[['selistpricehat', 'listpricehat']])
def get_all_data(forceUpdate=False):
    storedata = 'tmpsoldd.pandas'
    
    if os.path.exists(storedata) and not forceUpdate and not defaults['forceUpdate']:
        test = add_stata_modeled_to_df(pd.read_pickle(storedata))
        assert len(test.index) == len(test.index.unique())
        return add_stata_modeled_to_df( pd.read_pickle(storedata)), pd.read_pickle('all'+storedata)


    ff3,rf3 = extract_data_from_index_pages('current-manual')
    dfk3,df3 = process_features_to_final_dataframe(ff3, rf3)
    dfk3['sold'] = False

    ff2,rf2 = process_folder_of_details_pages('/home/cpbl/Dropbox/househunting/comparables4640/')
    dfk2,df2 = process_features_to_final_dataframe(ff2, rf2)
    dfk2['sold'] = False

    #paths['indexeshtml'] = 
    ff,rf =extract_data_from_index_pages('sold-manual')
    dfk,df = process_features_to_final_dataframe(ff, rf)
    dfk['sold'] = True


    for adf in [df,df2,df3,  dfk,dfk2, dfk3]:        adf.index.name='uid'

    dfk = pd.concat([dfk, dfk2,dfk3])
    dfk = dfk[~dfk.index.duplicated(keep='first')]
    df = pd.concat([df,df2,df3])
    df = df[~df.index.duplicated(keep='first')]

    
    assert len(dfk.index) == len(dfk.index.unique())
    assert len(df.index) == len(df.index.unique())
    
    dfk.to_pickle(storedata)
    df.to_pickle('all'+storedata)
    test = add_stata_modeled_to_df(pd.read_pickle(storedata))
    assert len(test.index)==len(test.index.unique())

    return add_stata_modeled_to_df(pd.read_pickle(storedata)), pd.read_pickle('all'+storedata)    


def statareg(latex):
    dfk, df = get_all_data()

    dfk.reset_index().to_stata(paths['working']+'allunits.dta')
    os.system('gzip -f {WP}allunits.dta'.format(WP=paths['working']))
    models = latex.str2models("""
*name:All units
reg listprice livingSpace  bedrooms bathrooms levels parkings firstfloor secondfloor thirdfloor  undivided 
reg listprice livingSpace indoorArea outdoorArea bedrooms bathrooms levels parkings firstfloor secondfloor thirdfloor  undivided 
*name:All units
reg listprice livingSpace indoorArea outdoorArea bedrooms bathrooms levels parkings firstfloor secondfloor thirdfloor  undivided  if sold==0
reg listprice livingSpace indoorArea outdoorArea bedrooms bathrooms levels parkings firstfloor secondfloor thirdfloor  undivided totalAnnualCosts if sold==0
*name:Undivided
*flag:undivided=yes
reg listprice livingSpace indoorArea outdoorArea bedrooms bathrooms levels parkings firstfloor secondfloor thirdfloor if undivided 
*flag:undivided=no
*name:Divided
reg listprice livingSpace indoorArea outdoorArea bedrooms bathrooms levels parkings firstfloor secondfloor thirdfloor  if undivided==0
""")
    models[0]['code']['after']="""
capture drop listpricehat selistpricehat
predict listpricehat
predict selistpricehat, stdp
"""+pst.stataSave(pst.WPdta('allunitsModeled'))
    outs= pst.stataLoad(pst.WPdta('allunits'))+"""
    """+ latex.regTable('duproprio',models, variableOrder=[
        'undivided'
'bedrooms', 
'bathrooms', 
'livingSpace', 
'indoorArea', 
'outdoorArea', 
'parkings', 
'firstfloor', 
'secondfloor', 
'thirdfloor', 
'levels', 
    ])


    if 'listpricehat' in dfk:
        sdf = dfk.join(df[['streetAddress']]).reset_index().dropna(subset=['listpricehat'])[['listprice','listpricehat','selistpricehat','bedrooms','livingSpace', 'uid','streetAddress','distance']].fillna('').sort_values('distance')

        sdf=sdf.query('listprice<950000 and listprice>=400000')

        sdf['bedrooms'] = sdf['bedrooms'].map(lambda ss: int(ss)*2*'~'+str(int(ss))) 
        for toint in ['listprice','listpricehat','selistpricehat','distance','livingSpace']:
            sdf[toint]= sdf[toint].astype(int).astype(str)
            
        from cpblUtilities.textables import dataframeWithLaTeXToTable
        dataframeWithLaTeXToTable(sdf.rename(columns={'listpricehat':'Modeled price', 'selistpricehat':'uncertainty','bedrooms':'BR'}), paths['output']+'modeledprices')
        """
        df,
        outfile,
        tableTitle=None,
        caption=None,
        label=None,
        footer=None,
        tableName=None,
        landscape=None,
        masterLatexFile=None,
        boldHeaders=False,
        boldFirstColumn=False,
        columnWidths=None,
        formatCodes=None, #'lc',
        formatString=None,
        hlines=False,
        pdfcrop=False)
        """
    return outs


#########################################################################################################
#=======================================================================================================#
#===================================  MAIN =============================================================#
#=======================================================================================================#
#=======================================================================================================#
#########################################################################################################
defaults={}
if __name__ == '__main__':
    # Docopt is a library for parsing command line arguments
    if 1:#try:
        # Parse arguments, use file docstring as a parameter definition
        arguments = docopt.docopt(__doc__)
        knownmodes = [aa for aa in arguments if not aa.startswith('-')]
        #if arguments['--serial']:
        #        defaults['server']['parallel']=False
        if arguments['--index-path']:
                paths['indexhtml']= arguments['--index-path']+'/'
        if arguments['--details-path']:
                paths['detailshtml']= arguments['--details-path']+'/'
        defaults['forceUpdate'] = arguments['--force-update'] == True
        runmode=''.join([ss*arguments[ss] for ss in knownmodes])
        runmode= None if not runmode else runmode
    # Handle invalid options
    else:#except docopt.DocoptExit as e:
        print e.message


    if runmode in ['details']:
        features, roomfeatures = process_folder_of_details_pages()

    if runmode in ['indexes']:
        extract_data_from_index_pages(paths['indexhtml'])
        
        ####use_index_pages_to_get_sold_property_pages(paths['indexhtml'])

        foowowo2
        
    if runmode in ['prep']:
        dfk,df = get_all_data(False)
        
    if runmode in ['plot']: # None:
        if paths.get('indexhtml',None) is None:
            paths['indexhtml'] = 'sold-manual'
        dfk,df = get_all_data(False)

        dfku = dfk[dfk.divided=="Undivided"]
        dfkd = dfk[dfk.divided=="Divided"]


        # Analysis

        # Difference in prices based only on area?
        fig,axs = plt.subplots(2)
        ax=axs[0]
        for dd in ['Divided','Undivided']:
            yy = dfk.query('divided=="{}"'.format(dd))
            ax.plot(yy.listprice/1000, yy.indoorArea, '.', label=dd)
        ax.set_xlabel(r'Asking price (k\$)')
        ax.set_ylabel('Indoor area (m$^2$, sum over rooms)')
        ax.set_ylim([0,250])
        ax=axs[1]
        for dd in ['Divided','Undivided']:
            yy = dfk.query('divided=="{}"'.format(dd))
            ax.plot(yy.listprice/1000, yy.livingSpace, '.', label=dd)
        ax.set_xlabel(r'Asking price (k\$)')
        ax.set_ylabel('Living space (m$^2$)')
        ax.legend()
        ax.set_ylim([0,250])
        plt.savefig('price-div.pdf')

        # sum of area matchs living space?
        fig,axs = plt.subplots(2)
        ax=axs[0]
        dfk[['indoorArea','livingSpace']].plot.scatter('indoorArea','livingSpace', ax=ax)
        ax.plot(ax.get_xlim(), ax.get_xlim(), 'k', zorder =-10)
        ax.set_xlabel('Indoor area (m$^2$)')
        ax.set_ylabel('Living space (m$^2$)')
        ax=axs[1]
        (dfk.livingSpace/dfk.indoorArea).hist(bins=100, ax=ax)
        ax.set_xlim([0,2])
        ax.set_xlabel('Ratio of habitable area / sum(indoor room areas)')
        plt.savefig('area-agreement.pdf')

        fig,axs = plt.subplots(2)
        #plt.style.use('seaborn-deep')
        bins = np.linspace(200, 800, 31)
        ax=axs[0]
        ax.hist([dfku.ppsqft.dropna(), dfkd.ppsqft.dropna()], bins, label=['Undivided', 'Divided'])
        ax.legend(loc='upper right')
        ax.set_xlabel('Price/square-foot (habitable area)')
        ax=axs[1]
        ax.hist([dfku.ppsqft_r.dropna(), dfkd.ppsqft_r.dropna()], bins, label=['Undivided', 'Divided'])
        ax.legend(loc='upper right')
        ax.set_xlabel('Price/square-foot (indoor rooms)')
        plt.savefig('ppsqft-div.pdf')


        fig,ax = plt.subplots(1)
        bins = np.linspace(200000, 800000, 31)
        for br in [2,3,4]:
            #ax.hist([dfk.query('bedrooms=={}'.format(br)).listprice.values for br in [2,3,4]], bins, label=['2','3','4'])
            ax.hist(dfk.query('bedrooms=={}'.format(br)).listprice.values , bins, label=str(br), alpha=.3)#['2','3','4'])
        ax.legend(loc='upper right')
        ax.set_xlabel('Asking price')
        plt.savefig('price-br.pdf')


        plt.show()
        #dfk.groupby('divided')['ppsqft'].hist(ax=ax)
        #ax.legend()


        


        dodoo
        foiu
    elif runmode in ['dev']:
        foo
    elif runmode in ['convert_areas']:
        import sys,re
        while (1):
            print(' Copy and paste the "Room dimensions" section from a Duproprio page. Then type Ctrl-D to get the sum.')
            complete_inout = sys.stdin.read()

            print('\n\n\n\n\n\n')
            buffer =''
            rooms=[]
            for aline in complete_inout.split('\n'):
                if 'm)' not in aline:
                    buffer= aline
                    continue
                mm= re.findall('([,.0123456789]*) m x ([,.0123456789]*) m' , aline)
                x,y =  float(mm[0][0].replace(',','.')) ,  float( mm[0][1].replace(',','.'))
                print '{}\t{}\t{}\t\t{}'.format(buffer, x,y,x*y)
                buffer=''
                rooms+= [x*y]


                print '\nTOTAL\t\t\t\t{}'.format(sum(rooms))


    elif runmode in ['photos']:
        get_photos_for_folder_of_details_pages()
    elif runmode in ['stata','prep']:
        sVersion,rVersion,dVersion = 'A','1','d'        
        pst.runBatchSet(sVersion,rVersion,[
            statareg,
            ],dVersion=dVersion)
        
        dfk,df = get_all_data(False)
                
