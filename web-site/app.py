from warnings import simplefilter
from os       import system
from shelve   import open
from json     import dumps, loads

simplefilter(action = 'ignore', category = FutureWarning)
system('clear')

import pandas_datareader                as     dr
import pandas                           as     pd
import numpy                            as     np

from   dash                             import Dash
from   dash.dependencies                import State as S, Input as I, Output as O
from   dash_html_components             import Div, P, B, Span, Label, Img, H1, H2, Hr, A, Img, Link, Iframe
from   dash_core_components             import Graph, Loading, Dropdown, DatePickerRange, RangeSlider, Markdown
from   dash_bootstrap_components        import Container, Row, Col, Tooltip, Collapse, Modal, ModalHeader, ModalBody, ModalFooter, Navbar, NavbarBrand, NavbarToggler, Jumbotron, Card, CardBody, Alert, Button, DropdownMenu, DropdownMenuItem
from   dash_bootstrap_components.themes import BOOTSTRAP, CYBORG, CERULEAN, DARKLY, FLATLY
from   plotly.graph_objects             import Figure, Layout, Choropleth, Scatter, Sunburst, Bar, Table
from   datetime                         import date, datetime

import plotly.express    as px # high level api
import plotly.graph_objs as go # low  level api

T = True
F = False

BG_PAPER  = 'rgb(255,255,255)'
BG_PLOT   = 'rgb(255,255,255)'
BG_CHART  = 'rgb(255,  0,255)'
BG_LEGEND = 'rgb(255,255,   )'
FG_TEXT   = 'rgb(255,255,255)'

class memo :

    def __init__(self, func) :
        self.func = func
        self.memo = open(f'{func.__name__}')

    def __call__(self, *args) :

        nick = repr(args)

        if  not nick in self.memo :
            self.memo[nick] = self.func(*args)
            print(f'memoize : executed : {self.func.__name__} : {args}')
        else                      :
            print(f'memoize : recalled : {self.func.__name__} : {args}')

        return self.memo[nick]

    def __del__(self) :

        self.memo.close()

    def memoize(f) :
        memo = {}
        def wrapper(x) :
            if  x not in memo :
                memo[x] = f(x)
            return memo[x]
        return wrapper

THEME = FLATLY

def fetch(table, field, value, opera = 'eq') :

    key = f'{table}_{field}_{value}_{opera}'

    tab = data[table]
    col = tab[field]

    if  not key in data['cache'] :

        data['cache'][key] = tab[col == value] if opera == 'eq' else \
                            tab[col != value] if opera == 'ne' else \
                            tab[col in value] if opera == 'in' else \
                            tab[col         ]

    return data['cache'][key]

def upFigure(traces, layout, margin = (60,10,10,10)) :

    figure = Figure(data = traces, layout = layout)


    if  margin :
        figure.layout.margin.autoexpand = False
        figure.layout.margin.pad        = 0
        figure.layout.margin.t          = margin[0]
        figure.layout.margin.b          = margin[1]
        figure.layout.margin.l          = margin[2]
        figure.layout.margin.r          = margin[3]

    figure.layout.title.font.size   = 24

    figure.layout.paper_bgcolor     = BG_PAPER
    figure.layout.plot_bgcolor      = BG_PLOT

    return figure

def makeCard(plot, view, hide = False) :

    layout = \
    {
        'title'            : f'<b>{plot} of {view}</b>',

        'xaxis_showgrid'   : False,
        'xaxis_zeroline'   : False,
        'xaxis_visible'    : False,
        'yaxis_showgrid'   : False,
        'yaxis_zeroline'   : False,
        'yaxis_visible'    : False,

        'paper_bgcolor'    : BG_PAPER,
        'plot_bgcolor'     : BG_PLOT,
    }

    config = \
    {
        'displaylogo' : False,
        'showTips'    : True,
        'frameMargins' : 0,
    }

    style = \
    {
        'background' : 'blue'
    }

    wrap = plot.replace('plot', 'wrap')
    card = plot.replace('plot', 'card')
    mask = plot.replace('plot', 'mask')
    load = plot.replace('plot', 'load')

    figure = upFigure(traces = [], layout = layout)

    return  Div(id = wrap, className = 'wrap', children =
            [
                Card(id = card, className = f'{view}', children =
                [
                    Div(id = mask, className = 'mask', hidden = False, children =
                    [
                        Loading(id = load, className = 'load', children =
                        [
                            Graph(id = plot, figure = figure, config = config, style = style)
                        ]),
                    ])
                ])
            ])

def makeRoot_MandABar(hide = T) :

    return Div(
        Container(
            [
                Col([
                        Label('activity window'),
                        Div(id          = 'manda-date-div',
                            children    = DatePickerRange(
                            id          = 'manda-date',
                            )
                        ),
                        Tooltip(
                            children    = 'Select Date Range to Filter Stock Market Performance and M & A History',
                            id          = 'manda-date-tip',
                            target      = 'manda-date-div',
                            placement   = 'right')
                    ],
                    className = 'dis nop r1p',
                    width     = 6,
                    style     = {'display' : 'none'}
                ),
                Col([
                        Label('activity filter'),
                        Div(id          = 'manda-type-div',
                            children    = Dropdown(
                            id          = 'manda-type',
                            placeholder = 'Select M & A Type',
                            options     = [{'label' : v, 'value' : v} for v in data['tlist']],
                            value       = 'Merger/Acquisition')
                        ),
                        Tooltip(
                            children    = 'Select M & A Type to Filter M & A History',
                            id          = 'manda-type-tip',
                            target      = 'manda-type-div',
                            placement   = 'left')
                    ],
                    className = 'dis nop',
                    width     = 3,
                ),
            ],
            id    = 'manda-tool',
            fluid = True,
            style = {'padding' : '0rem'}
        ),  id    = 'manda-wrap', hidden = hide)

def makeRoot_RisksBar(hide = F) :

    return Div(
        Container(
            [
                Col([
                        Label('operational region(s)'),
                        Div(id          = 'risks-area-div',
                            children    = Dropdown(
                            id          = 'risks-area',
                            placeholder = 'Select Operating Area(s)',
                            options     = [{'label': v, 'value': v} for v in data['alist']],
                            value       = data['alist'][0:],
                            multi       = True)
                        ),
                        Tooltip(
                            children    = 'Select Operating Area(s) to See Details on the Geographical Footprint Map',
                            id          = 'risks-area-tip',
                            target      = 'risks-area-div',
                            placement   = 'right')
                    ],
                    className = 'dis nop r1p',
                    width     = 6,
                    style     = {'display' : 'none'}
                ),
                Col([
                        Label('operational footprint'),
                        Div(id          = 'risks-foot-div',
                            children    = Dropdown(
                            id          = 'risks-foot',
                            placeholder = 'Select Corporate Footprint',
                            options     = [{'label' : k, 'value' : v} for k, v in data['fdict'].items()],
                            value       = 'subsidiaries')
                        ),
                        Tooltip(
                            children    = 'Select Footprint Metric to See Details on the Geographical Footprint Map',
                            id          = 'risks-foot-tip',
                            target      = 'risks-foot-div',
                            placement   = 'left')
                    ],
                    className = 'dis nop',
                    width     = 3,
                )
            ],
            id    = 'risks-tool',
            fluid = True,
            style = {'padding' : '0rem'}
        ),  id    = 'risks-wrap', hidden = hide)

def makeRoot_ToolsBar() :

    _ = \
    [
        Col([
                NavbarBrand(
                    [
                        Img(
                            src    = 'assets/logo.gif',
                            height = '36px',
                            width  = '144px',
                            className = 'nop nom'
                        ),
                    ],
                    className = 'nop nom',
                ),
            ],
            className = 'dis nop',
            width     = 2,
        ),

        Col([   Label('corporation'),
                Div(id          = 'tools-corp-div',
                    children    = Dropdown(
                    id          = 'tools-corp',
                    placeholder = 'Select Corporation',
                    options     = [{'label' : f'{k} ({v})', 'value' : k} for k, v in data['hdict'].items()],
                    value       = 'Apple',
                    searchable  = True)
                ),
                Tooltip(
                    children    = 'Select Corporation to See Stock Market Performance and M & A History',
                    id          = 'tools-corp-tip',
                    target      = 'tools-corp-div',
                    placement   = 'right')
            ],
            className = 'dis nop r1p',
            width     = 2,

        ),

        Col([
                makeRoot_MandABar(),
                makeRoot_RisksBar()
            ],
            className = 'nop r1p',
            width = 6,
        ),

        Col([
                Label('current view'),
                DropdownMenu(
                    [
                        DropdownMenuItem('Mergers & Acquisitions', id = 'manda-pick'),
                        DropdownMenuItem('Corporate Global Risks', id = 'risks-pick'),
                    ],
                    id          = 'views-pick',
                    label       = 'Corporate Global Risks',
                    color       = 'primary',
                    style       = {'lineHeight' : '22px'},
                    className   = ''
                ),
                Tooltip(
                    children    = 'Select Activity View',
                    id          = 'views-pick-tip',
                    target      = 'views-pick',
                    placement   = 'left')
            ],
            className = 'dis nop',
            width     = 2
        )
    ]

    return Navbar(
            id        = 'extra-bar',
            className = 'rounded v1m',
            children  = Container(
                            _,
                            className = 'nom nop',
                            fluid     = True
                        ))

def makeRoot_Contents() :

    return Container(
                [
                    Div(id = 'store', hidden = True),
                    Modal(
                        [
                            ModalHeader(
                                [
                                    B('Welcome to the Global Risk Management Visualization')
                                ]
                            ),
                            ModalBody(
                                P(
                                [
                                    B('Global risk data for everyone'),
                                    P('It is frustrating when you can nott make a qualified assessment on the global risk position of a multinational corporation due to the enormous complexity of its operations. The global risk visualization addresses that challenge by offering the most intuitive options to interact with corporate risk data all in one place.'),
                                    B('Valuable insights through disparate data connections'),
                                    P('By connecting financial performance, global footprint, and national risk data for over 100 countries, the visualization helps you get a better sense of the political, economic and financial risks facing these organizations.'),
                                    Hr(),
                                    B('Target Audience:'),
                                    P('Investors, Students, Corporate Legal and Researchers'),
                                    Hr(),
                                    B('Data Sources:'),
                                    P('Yahoo Finance, PRS Global Risk Dataset, Orbis Global Footprint Dataset, and SAP Capital IQ M&A Dataset'),
                                    Hr(),
                                    B('Team:'),
                                    P('Jay Venkata, Keith Wertsching, and Pri Nonis'),
                                    Hr(),
                                    Iframe(src='https://zoom.us/rec/play/vJYqcu-grzM3SIWW5ASDB_IqW9W9La6sgyke8qdYyx63VSZRYVCuMuYRMAI0fIKQL0LNUUuXXcgTFVo', width = 1106, height = 650)
                                ]
                                )
                            ),
                            ModalFooter(
                                Button('Close', id = 'close', className = 'ml-auto')
                            ),
                        ],
                        id      = 'modal',
                        size    = 'xl',
                        is_open = True
                    ),

                    makeRoot_ToolsBar(),

                    Row([
                            Col(
                                makeCard('stock-plot', 'manda', hide = False),
                                width = 12,
                            ),
                            # Col(
                            #     makeCard('table-plot', 'manda', hide = False),
                            #     width = 3
                            # ),
                        ],
                        className = 'manda'
                    ),

                    Row([
                            Col(
                                makeCard('world-plot', 'risks', hide = True),
                                width = 6,
                                style = {'paddingRight' : '0rem'}
                            ),
                            Col(
                                makeCard('total-plot', 'risks', hide = True),
                                width = 6
                            ),
                        ],
                        className = 'risks t1p'
                    ),
                    Row([
                            Col(
                                makeCard('burst-plot', 'risks', hide = True),
                                width = 3,
                                style = {'paddingRight' : '0rem'}
                            ),
                            Col(
                                makeCard('trend-plot', 'risks', hide = True),
                                width = 9
                            ),
                        ],
                        className = 'risks v1p'
                    ),
                ],
                fluid = True,
                style =
                {
                }
            )

def makeRoot() :

    return  Div([
                    makeRoot_Contents()
                ]
            )

    return roo

def makeApp() :

    dash = Dash(
        __name__,
        meta_tags =
        [
            {
                'name'    : 'viewport',
                'content' : 'width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no'
            }
        ],
        external_stylesheets = [THEME]
    )

    dash.title  = 'Global Risk Overview for Corporations'
    dash.head   = \
    [
        Link(href = 'assets/globe.gif', rel = 'icon')
    ]
    dash.layout = makeRoot()

    return dash

def loadCSV() :

    print('Loading Data : Please Wait...', end = '')

    risks = pd.read_csv(f'data/risks.csv.gz', compression = 'gzip', parse_dates = ['date'               ])
    manda = pd.read_csv(f'data/manda.csv'   ,                       parse_dates = ['announced', 'closed'])
    scope = pd.read_csv(f'data/scope.csv'   ,                       parse_dates = [                     ])

    final = risks.date.max()
    rlast = risks[risks.date.eq(final)]

    rlist = risks.nick.unique().tolist()
    clist = scope.corporation.unique().tolist()
    alist = scope.region.unique().tolist()
    tlist = manda.type.unique().tolist()

    fdict = \
    {
        'Subsidiaries' : 'subsidiaries',
        'Employees'    : 'employees',
        'Assets'       : 'assets',
        'Shareholders' : 'funds',
        'Revenue'      : 'revenue',
        'Profits'      : 'profits',
    }

    cnmap = dict(zip(scope.code, scope.country))

    cdict = \
    {
        'Apple'     : 'AAPL',
        'Boeing'    : 'BA',
        'DuPont'    : 'DD',
        'Exxon'     : 'XOM',
        'Walmart'   : 'WMT',
        'Dow'       : '^DJI'
    }

    hdict = \
    {
        'Apple'     : 'USA',
        'Boeing'    : 'USA',
        'DuPont'    : 'USA',
        'Exxon'     : 'USA',
        'Walmart'   : 'USA',
    }

    cache = {}

    print('Done.')

    return {
        'risks' : risks,
        'scope' : scope,
        'manda' : manda,

        'final' : final,
        'rlast' : rlast,

        'rlist' : rlist,
        'clist' : clist,
        'alist' : alist,
        'tlist' : tlist,

        'cdict' : cdict,
        'fdict' : fdict,
        'hdict' : hdict,

        'cnmap' : cnmap,

        'cache' : cache
    }

data = loadCSV()
dash = makeApp()

def waitMake(note = 'Select', title = 'The Title') :

    layout = \
    {
        'title'          : f'<b>{title}</b>',

        'xaxis_showgrid' : False,
        'xaxis_zeroline' : False,
        'xaxis_visible'  : False,
        'yaxis_showgrid' : False,
        'yaxis_zeroline' : False,
        'yaxis_visible'  : False,

        'annotations'    : [{
            'text'       : f'<span style="color:firebrick;font-size:32px"><b>{note}</b></span>',
            'showarrow'  : False
            }
        ],

        'paper_bgcolor'  : BG_PAPER,
        'plot_bgcolor'   : BG_PLOT,
    }

    return [upFigure(traces = [], layout = layout)]

@dash.callback(
    [
        O('modal', 'is_open')
    ],
    [
        I('close', 'n_clicks'),
    ]
)
def makePlot_Stock(close) :

    if  close : return [False]
    else      : return [ True]

@memo
def yahoo(stock, start = date(2000, 4, 1), end = date(2020, 4, 1)) :

    return dr.DataReader(stock, 'yahoo', start, end)

@dash.callback(
    [
        O('stock-plot', 'figure')
    ],
    [
        I('tools-corp', 'value'),
        I('manda-type', 'value'),
    ]
)
def makePlot_Stock(corp, ftype) :

    if  not all([corp]) :
        return waitMake('Select Corp and Filter Type from Downdowns', 'M & A History and Relative Stock Market Performance')

    sdate = datetime(2000, 4, 1)
    edate = datetime(2020, 4, 1)

    manda = data['manda']
    stock = yahoo(data['cdict'][corp ], sdate, edate)
    bench = yahoo(data['cdict']['Dow'], sdate, edate)

    frame = pd.DataFrame()

    fdrop = 'Patent'
    space = ' '
    nsize = 'Undisclosed'

    tempo = manda[ manda.corporation.eq(      corp)&
                   manda.type.eq(            ftype)&
                   manda.announced.ge(       sdate)&
                   manda.announced.le(       edate)]

    # tempo = manda[ manda.corporation.eq(      corp)&
    #                manda.type.eq(            ftype)&
    #                manda.buyer.str.contains(  corp)&
    #               ~manda.target.str.contains(fdrop)&
    #                manda.announced.ge(       sdate)&
    #                manda.announced.le(       edate)]

    frame['announce'] = tempo.announced
    frame['relative'] = tempo.ammount / tempo.ammount.max() * 40
    frame['relative'] = frame.relative.mask(frame.relative < 5, 5)

    frame['hovering'] = tempo.apply(lambda x : f'Acquired {space.join(x.target.split(space)[:7])}<br>for {round(x.ammount/1000, 3) if x.ammount else nsize} Million Dollars', axis = 1)
    frame['disclose'] = tempo.apply(lambda x : 'circle'   if x.ammount else 'circle-open', axis = 1)
    frame['palettes'] = tempo.apply(lambda x : 'seagreen' if x.ammount else 'green',       axis = 1)

    frame['position'] = -10

    stock['percent' ] = stock.Close / stock.Close[0]
    bench['percent' ] = bench.Close / bench.Close[0]

    trace0 = \
    Scatter(
        name           = corp,
        x              = stock.index,
        y              = stock.percent,
        mode           = 'lines',
        line_shape     = 'spline',
        line_smoothing = 1.3,
        line_color     = 'rgba(78,193,255,0.95)',
        line_width     = 4,
        stackgroup     = 'one'
    )

    trace1 = \
    Scatter(
        name           = 'Dow Jones',
        x              = bench.index,
        y              = bench.percent,
        mode           = 'lines',
        line_color     = 'rgba(63,1,125,0.8)',
        line_width     = 3,
        line_smoothing = 1.3,
    )

    trace3 = \
    Scatter(
        name          = ftype,
        x             = frame.announce,
        y             = frame.position,
        text          = frame.hovering,
        marker_size   = frame.relative,
        marker_symbol = frame.disclose,
        marker_color  = frame.palettes,
        marker_line_color = 'green',
        mode          = 'markers'
    )

    trace2 = \
    Scatter(
        name           = 'Timeline',
        x              = [sdate, edate],
        y              = [  -10,   -10],
        mode           = 'lines',
        line_color     = 'black',
        line_width     = 1,
        line_smoothing = 1.3,
    )

    layout = \
    Layout(
        title         = f'<b>M & A History and Relative Stock Market Performance of {corp}</b>',
        yaxis_title   = f'<b>Cumulative Return<br>(Multiple of Starting Value)<b>',
        xaxis_title   = '<b>Dates<b>',
        yaxis_ticksuffix = 'x'
    )

    return [upFigure(traces = [trace0, trace1, trace2, trace3], layout = layout, margin = (60,40,60,200))]

@dash.callback(
    [
        O('total-plot', 'figure')
    ],
    [
        I('world-plot', 'clickData')
    ]
)
def makePlot_Total(code) :

    if  not all([code]) :
        return waitMake('Select Country from Map', 'Prominent Risks in Selected Country vs. the Global Median')

    code  = code['points'].pop()['location'] if code else ''
    name  = data['cnmap' ][code]             if code else 'Country'

    risks = data['risks']
    risks = risks[risks.nick != 'Total']
    final = risks.date.max()

    latest_country = risks[risks.date.eq(final)&
                           risks.code.eq(code)]

    median_country = latest_country.groupby('nick').perc.median().sort_values(ascending = False)

    latest_world   = risks[risks.date.eq( final)&
                           risks.nick.isin(median_country.index.tolist()[:10])]

    median_world   =   latest_world.groupby('nick').perc.median()

    x0     = median_country.index.tolist()[:10]
    y0     = median_country.values.tolist()[:10]
    x1     = median_world.index.tolist()
    y1     = median_world.values.tolist()

    trace0 = \
    Bar(
        x      = y0,
        y      = x0,
        marker =
        {
            'color' : 'rgba(50, 171, 96, 0.6)',
            'line'  :
            {
                'color' : 'rgba(50, 171, 96, 1.0)',
                'width' : 1
            }
        },

        name        = f'{name}',
        orientation = 'h',
    )

    trace1 = \
    Bar(
        x      = y1,
        y      = x1,
        marker =
        {
            'color' : 'rgba(170, 131, 126, 0.6)',
            'line'  :
            {
                'color' : 'rgba(70, 71, 196, 1.0)',
                'width' : 1
            }
        },

        name        = 'Global Median',
        orientation = 'h',
    )

    layout = \
    Layout(
        title                = f'<b>Prominent Risks in {name} vs. the Global Median<b>',
        yaxis_showgrid       = False,
        yaxis_showline       = False,
        yaxis_showticklabels = True,

        xaxis_title          = '<b>Relative Risk Exposure<b>',
        xaxis_zeroline       = False,
        xaxis_showline       = False,
        xaxis_showticklabels = True,

        barmode              = 'group',
    )

    return [upFigure(traces = [trace0, trace1], layout = layout, margin = (60,40,140,160))]

@dash.callback(
    [
        O('world-plot', 'figure')
    ],
    [
        I('tools-corp', 'value'),
        I('risks-area', 'value'),
        I('risks-foot', 'value'),
    ]
)
def makeWorldPlot(corp, area, foot) :

    if  not all([corp, area, foot]) :
        return waitMake('Select Corp, Area(s), and Footprint Measure from Dropdown(s)', 'Geographical Footprint')

    scope = data['scope']
    cnmap = data['cnmap']

    tempo = scope[(scope.corporation == corp)&
                  (scope.region.isin(area)  )].groupby('code').sum()

    tempo['country'] = tempo.index.map(cnmap)
    tempo['scaled' ] = tempo[foot].map(np.log10)
    tempo['hover'  ] = tempo.apply(lambda x : f'{x.country} {foot.title()} : {x[foot]}', axis = 1)

    trace0 = \
    Choropleth(
        locations  = tempo.index,
        z          = tempo.scaled,
        text       = tempo.hover,
        showscale  = False,
        colorscale = 'purples',

        colorbar_bgcolor            = f'rgb(255, 255, 255)',
        colorbar_tickprefix         = f'10^',
        colorbar_title              = f'<b># of {foot.title()}</b><br>logarithmic scale',
    )

    layout = \
    Layout(
        title                       = f'<b>Geographical Footprint of {corp} by {foot.title()}<b>',
        geo_projection_type         = 'natural earth',
        geo_projection_rotation_lon = 0,
        geo_projection_rotation_lat = 0,
        geo_projection_scale        = 1,
        geo_showocean               = True,
        geo_showland                = True,
        geo_showlakes               = True,
        geo_lakecolor               = 'rgb(51, 193, 255)',
        geo_oceancolor              = 'rgb(51, 193, 255)',
        geo_bgcolor                 = 'rgb(255, 255, 255)',

        paper_bgcolor               = BG_PAPER,
        plot_bgcolor                = BG_PLOT,
    )

    return [upFigure(traces = [trace0], layout = layout, margin = (60,0,0,0))]

@dash.callback(
    [
        O('burst-plot', 'figure')
    ],
    [
        I('world-plot', 'clickData')
    ]
)
def makeBurstPlot(code) :

    if  not code :
        return waitMake('Select Country from Map', 'Current Risk Breakdown in Selected Country')

    code   = code['points'].pop()['location'] if code else ''
    name   = data['cnmap' ][code]             if code else 'Country'

    tempo  = data['rlast'][data['rlast'].code.eq(code)]

    nick = tempo.nick.tolist()
    root = tempo.root.tolist()
    rate = tempo.rate.tolist()
    rati = tempo.rati.tolist()
    perc = tempo.perc.tolist()

    parents = []
    values  = []
    labels  = []

    for n in range(41):
        print(nick[n], root[n], rate[n], perc[n])
        if  nick[n] != 'Total' and rati[n] > 0:
            parents += [root[n]] if root[n] != 'Total' else [None]
            values  += [rati[n]]
            labels  += [nick[n]]

    trace0 = \
    Sunburst(
        labels       = labels,
        parents      = parents,
        values       = values,
        branchvalues = 'total',
    )

    layout = \
    Layout(
        title       = f'<b>Current Risk Breakdown in {name}</b>',
        xaxis_title = f'<b>Unit Risk Exposure <a href="https://docs.google.com/spreadsheets/d/1HSR3GIjPgz6KsU2DWNKDiuk5BxqXx1McbkTSyv3ZVNo">Details</a></b>',
        annotations = [
                dict(
                    x         = 2,
                    y         = 5,
                    xref      = "x",
                    yref      = "y",
                    text      = "",
                    showarrow = False,
                    arrowhead = 0,
                    ax        = 0,
                    ay        = -40
                )]
    )

    return [upFigure(traces = [trace0], layout = layout, margin = (60,0,10,10))]

@dash.callback(
    [
        O('trend-plot', 'figure')
    ],
    [
        I('world-plot', 'clickData'),
        I('burst-plot', 'clickData')
    ]
)
def makeTrendPlot(code, root) :

    if  not all([code, root]) :
        return waitMake('Select Country from Map, and Risk Category from Sunburst', 'Historical Stream of Selected Risk in Selected Country')

    code   = code['points'].pop()['location'] if code else ''
    root   = root['points'].pop()['label'   ] if root else ''
    name   = data['cnmap' ][code]             if code else 'Country'

    risks  = data['risks']

    tempo  = risks[(risks.code == code)]

    traces = []

    nicks  = risks[risks.root == root].nick.unique().tolist()
    nicks  = risks[risks.nick == root].nick.unique().tolist() if not nicks else nicks

    color  = ['#6a3f39','#a87c79','#d6b3b1','#d8c3b2','#6a3f39','#f0e6de','#a87c79','#d6b3b1','#6a3f39','#f2f5fa']

    for n, nick in enumerate(nicks[:10]) :

        x = tempo[tempo.nick == nick].date.tolist()
        y = tempo[tempo.nick == nick].perc.tolist()

        c = color[n]

        t = \
        {
            'fill' : 'tonexty',
            'line' :
            {
        #       'color' : c,
                'width' : 0,
                'shape' : 'spline',
            },
            'mode'      : 'lines',
            'type'      : 'scatter',
        #   'fillcolor' : c,
            'name'      : nick,
            'x'         : x,
            'y'         : y,
            'stackgroup': 'one'
        }

        traces.append(t)

    river   = pd.read_csv('data/river.csv')
    traces_ = []

    for n, c in enumerate(color) :
        x = river[f'x'    ]
        y = river[f'y_{n}']

        t = \
        {
            'fill' : 'tonexty',
            'line' :
            {
            #   'color' : c,
                'width' : 0,
                'shape' : 'spline',
            },
            'mode'      : 'lines',
            'type'      : 'scatter',
          # 'fillcolor' : c,
            'x'         : x,
            'y'         : y,
        }

        traces_.append(t)

    layout = \
    Layout(
        title         = f'<b>Historical Stream of {root} Risk in {name}<b>',
        yaxis_title   = '<b>Relative Risk Exposure<b>',
        xaxis_title   = '<b>Dates<b>',
    )

    return [upFigure(traces = traces, layout = layout, margin = (60,40,40,200))]

@dash.callback(
    [
        O('store'     , 'children'),
    ],
    [
        I('manda-pick', 'n_clicks'),
        I('risks-pick', 'n_clicks'),
    ],
    [
        S('store'     , 'children')
    ]
)
def postStore(manda, risks, store) :

    store = loads(store or '{"manda" : 0, "risks" : 0}')
    manda = manda or 0
    risks = risks or 0

    if  manda > store['manda'] and store['manda'] <= store['risks'] : store['manda'] += 1
    if  risks > store['risks'] and store['risks'] <= store['manda'] : store['risks'] += 1

    print(f'postStore : {store}')

    return [dumps(store)]

@dash.callback(
    [
        O('manda-wrap', 'hidden'  ),
        O('stock-mask', 'hidden'  ),

        O('risks-wrap', 'hidden'  ),
      # O('world-mask', 'hidden'  ),
      # O('total-mask', 'hidden'  ),
      # O('burst-mask', 'hidden'  ),
      # O('trend-mask', 'hidden'  ),

        O('views-pick', 'label'   ),
    ],
    [
        I('store'     , 'children'),
    ],
    [
        S('views-pick', 'label')
    ]
)
def pickView(store, label) :

    store = loads(store or '{"manda" : 0, "risks" : 0}')

    if  store['risks'] > store['manda'] : print('pickView Risks View') ; return [T, T, F] + ['Corporate Global Risks']
    else                                : print('pickView MandA View') ; return [F, F, T] + ['Mergers & Acquisitions']

def main() :

    dash.run_server(debug = True, host = '0.0.0.0', port = 80)

if  __name__ == '__main__' :

    main()
