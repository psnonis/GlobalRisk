from warnings import simplefilter
from os       import system
from shelve   import open

simplefilter(action = 'ignore', category = FutureWarning)
system('clear')

import pandas_datareader                as     dr
import pandas                           as     pd
import numpy                            as     np

from   dash                             import Dash
from   dash.dependencies                import State, Input as I, Output as O
from   dash_html_components             import Div, P, B, Span, Label, Img, H1, H2, Hr, A, Img, Link
from   dash_core_components             import Graph, Loading, Dropdown, DatePickerRange, RangeSlider
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

def upFigure(traces, layout, margin = (60,10,10,10)) :

    figure = Figure(data = traces, layout = layout)

    figure.layout.margin.autoexpand = False
    figure.layout.margin.pad        = 0
    figure.layout.margin.t          = margin[0]
    figure.layout.margin.b          = margin[1]
    figure.layout.margin.l          = margin[2]
    figure.layout.margin.r          = margin[2]

    figure.layout.title.font.size   = 24

    figure.layout.paper_bgcolor     = BG_PAPER
    figure.layout.plot_bgcolor      = BG_PLOT

    return figure

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
                            placeholder = 'Select M&A Type',
                            options     = [{'label' : v, 'value' : v} for v in data['tlist']],
                            value       = 'Merger/Acquisition')
                        ),
                        Tooltip(
                            children    = 'Select M&A Type to Filter M&A History',
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
                                    B('Target Audience: Investors, Students, Corporate Legal and Researchers')
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

    scope = pd.read_csv(f'data/scope.csv')
    risks = pd.read_csv(f'data/risks.csv', parse_dates = ['date'])
    manda = pd.read_csv(f'data/manda.csv', parse_dates = ['announced', 'closed'])

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
    ]
)
def makePlot_Stock(corp) :

    if  not all([corp]) :
        return waitMake('Select Corp and Filter Type from Downdowns', 'M & A History and Relative Stock Market Performance')

    sdate = datetime(2000, 4, 1)
    edate = datetime(2020, 4, 1)

    manda = data['manda']
    stock = yahoo(data['cdict'][corp ], sdate, edate)
    bench = yahoo(data['cdict']['Dow'], sdate, edate)

    frame = pd.DataFrame()

    ftype = 'Merger/Acquisition'
    fdrop = 'Patent'
    space = ' '

    tempo = manda[ manda.corporation.eq(     corp)&
                   manda.type.eq(           ftype)&
                   manda.buyer.str.contains( corp)&
                  ~manda.target.str.contains(fdrop)&
                   manda.announced.ge(      sdate)&
                   manda.announced.le(      edate)]

    frame['announce'] = tempo.announced
    frame['relative'] = tempo.ammount / tempo.ammount.max() * 40
    frame['relative'] = frame.relative.mask(frame.relative < 5, 5)

    frame['hovering'] = tempo.apply(lambda x : f'{space.join(x.target.split(space)[:7])}', axis = 1)
    frame['disclose'] = tempo.apply(lambda x : 'circle'   if x.ammount else 'circle-open', axis = 1)
    frame['palettes'] = tempo.apply(lambda x : 'seagreen' if x.ammount else 'green',       axis = 1)

    frame['position'] = -10

    stock['percent'] = stock.Close / stock.Close[0]
    bench['percent'] = bench.Close / bench.Close[0]

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

    trace2 = \
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

    layout = \
    Layout(
        title       = f'<b>M & A History and Relative Stock Market Performance of {corp}</b>',
        xaxis       = dict(
        ),
        yaxis       = dict(

        ),
    )

    return [upFigure(traces = [trace0, trace1, trace2], layout = layout, margin=(60,40,40,1000))]

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
        return waitMake('Select Country from Map', 'Prominent Risks')

    code  = code['points'].pop()['location'] if code else ''
    name  = data['cnmap' ][code]             if code else 'Country'

    risks = data['risks']
    final = risks.date.max()

    latest_country = risks[risks.date.eq(final)&
                           risks.code.eq(code)]

    median_country = latest_country.groupby('nick').percent.median().sort_values(ascending = False)

    latest_world   = risks[risks.date.eq( final)&
                           risks.nick.isin(median_country.index.tolist()[:10])]

    median_world   =   latest_world.groupby('nick').percent.median()

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

        xaxis_zeroline       = False,
        xaxis_showline       = False,
        xaxis_showticklabels = True,
        barmode              = 'group',
    )

    return [upFigure(traces = [trace0, trace1], layout = layout, margin = (60,40,140,140))]

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

        margin      =
        {
            't' : 60,
            'b' :  0,
            'r' :  0,
            'l' :  0
        },

        paper_bgcolor               = BG_PAPER,
        plot_bgcolor                = BG_PLOT,
    )

    return [upFigure(traces = [trace0], layout = layout)]

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
        return waitMake('Select Country from Map', 'Current Risk Breakdown')

    code   = code['points'].pop()['location'] if code else ''
    name   = data['cnmap' ][code]             if code else 'Country'

    tempo  = data['rlast'][data['rlast'].code == code]

    trace0 = \
    Sunburst(
        labels       = tempo.nick,
        parents      = tempo.root,
        values       = tempo.rate,
        branchvalues = 'total',
        # marker       = dict(
        #     colorscale = 'rdbu'
        # )
    )

    layout = \
    Layout(
        title       = f'<b>Current Risk Breakdown in {name}</b>',
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
        return waitMake('Select Country from Map, and Risk Category from Sunburst', 'Historical Risk Stream')

    code   = code['points'].pop()['location'] if code else ''
    root   = root['points'].pop()['label'   ] if root else ''
    name   =  data['cnmap' ][code]             if code else 'Country'

    risks  = data['risks']

    tempo  = risks[(risks.code == code)]

    traces = []

    nicks  = risks[risks.root == root].nick.unique().tolist()
    nicks  = risks[risks.nick == root].nick.unique().tolist() if not nicks else nicks

    for n, nick in enumerate(nicks) :

        x = tempo[tempo.nick == nick].date.tolist()
        y = tempo[tempo.nick == nick].percent.tolist()

        t = \
        {
            'fill' : 'tonexty',
            'line' :
            {
                'width' : 1
            },
            'mode' : 'lines',
            'type' : 'scatter',
            'name' : nick,
            'x'    : x,
            'y'    : y,
            'stackgroup' : 'one'
        }

        #if  n == 0 : t.pop('fill')

        traces.append(t)

  #  data = []
  #  colo = []

    river  = pd.read_csv('data/river.csv')

    trace1 = {
  # 'fill': 'tonexty',
    'line': {
        'color': '#6a3f39',
        'width': 0
    },
    'mode' : 'lines',
    'name' : '',
    'type' : 'scatter',
    'x'    : [1, 1.0140280561122244, 1.028056112224449, 1.0420841683366733, 1.0561122244488979, 1.0701402805611222, 1.0841683366733468, 1.0981963927855711, 1.1122244488977957, 1.12625250501002, 1.1402805611222444, 1.154308617234469, 1.1683366733466933, 1.182364729458918, 1.1963927855711423, 1.2104208416833666, 1.2244488977955912, 1.2384769539078158, 1.25250501002004, 1.2665330661322645, 1.280561122244489, 1.2945891783567134, 1.308617234468938, 1.3226452905811623, 1.3366733466933867, 1.3507014028056112, 1.3647294589178358, 1.3787575150300602, 1.3927855711422845, 1.406813627254509, 1.4208416833667334, 1.434869739478958, 1.4488977955911824, 1.4629258517034067, 1.4769539078156313, 1.4909819639278556, 1.5050100200400802, 1.5190380761523046, 1.533066132264529, 1.5470941883767535, 1.561122244488978, 1.5751503006012024, 1.5891783567134268, 1.6032064128256514, 1.6172344689378757, 1.6312625250501003, 1.6452905811623246, 1.659318637274549, 1.6733466933867736, 1.6873747494989981, 1.7014028056112225, 1.7154308617234468, 1.7294589178356714, 1.7434869739478958, 1.7575150300601203, 1.7715430861723447, 1.785571142284569, 1.7995991983967936, 1.8136272545090182, 1.8276553106212425, 1.8416833667334669, 1.8557114228456912, 1.8697394789579158, 1.8837675350701404, 1.8977955911823647, 1.911823647294589, 1.9258517034068137, 1.9398797595190382, 1.9539078156312626, 1.967935871743487, 1.9819639278557113, 1.9959919839679359, 2.0100200400801604, 2.024048096192385, 2.038076152304609, 2.052104208416834, 2.066132264529058, 2.0801603206412826, 2.094188376753507, 2.1082164328657313, 2.122244488977956, 2.13627254509018, 2.150300601202405, 2.164328657314629, 2.1783567134268536, 2.1923847695390783, 2.2064128256513027, 2.220440881763527, 2.2344689378757514, 2.2484969939879758, 2.2625250501002006, 2.276553106212425, 2.2905811623246493, 2.304609218436874, 2.318637274549098, 2.3326653306613228, 2.346693386773547, 2.3607214428857715, 2.3747494989979963, 2.38877755511022, 2.402805611222445, 2.4168336673346693, 2.4308617234468937, 2.4448897795591185, 2.458917835671343, 2.472945891783567, 2.4869739478957915, 2.501002004008016, 2.5150300601202407, 2.529058116232465, 2.5430861723446894, 2.557114228456914, 2.571142284569138, 2.585170340681363, 2.599198396793587, 2.6132264529058116, 2.6272545090180364, 2.6412825651302603, 2.655310621242485, 2.6693386773547094, 2.6833667334669338, 2.6973947895791586, 2.7114228456913825, 2.7254509018036073, 2.7394789579158316, 2.753507014028056, 2.7675350701402808, 2.781563126252505, 2.7955911823647295, 2.809619238476954, 2.823647294589178, 2.837675350701403, 2.8517034068136273, 2.8657314629258517, 2.8797595190380765, 2.8937875751503004, 2.907815631262525, 2.9218436873747495, 2.935871743486974, 2.9498997995991987, 2.9639278557114226, 2.9779559118236474, 2.9919839679358717, 3.006012024048096, 3.020040080160321, 3.0340681362725452, 3.0480961923847696, 3.062124248496994, 3.0761523046092183, 3.090180360721443, 3.1042084168336674, 3.118236472945892, 3.132264529058116, 3.1462925851703405, 3.1603206412825653, 3.1743486973947896, 3.188376753507014, 3.2024048096192383, 3.216432865731463, 3.2304609218436875, 3.244488977955912, 3.258517034068136, 3.2725450901803605, 3.2865731462925853, 3.3006012024048097, 3.314629258517034, 3.3286573146292584, 3.342685370741483, 3.3567134268537075, 3.370741482965932, 3.3847695390781563, 3.3987975951903806, 3.4128256513026054, 3.4268537074148298, 3.440881763527054, 3.4549098196392785, 3.468937875751503, 3.4829659318637276, 3.496993987975952, 3.5110220440881763, 3.5250501002004007, 3.5390781563126255, 3.55310621242485, 3.567134268537074, 3.5811623246492985, 3.595190380761523, 3.6092184368737477, 3.623246492985972, 3.6372745490981964, 3.6513026052104207, 3.6653306613226455, 3.67935871743487, 3.693386773547094, 3.7074148296593186, 3.721442885771543, 3.7354709418837677, 3.749498997995992, 3.7635270541082164, 3.7775551102204408, 3.7915831663326656, 3.80561122244489, 3.8196392785571143, 3.8336673346693386, 3.847695390781563, 3.8617234468937878, 3.875751503006012, 3.8897795591182365, 3.903807615230461, 3.9178356713426856, 3.93186372745491, 3.9458917835671343, 3.9599198396793587, 3.973947895791583, 3.987975951903808, 4.002004008016032, 4.0160320641282565, 4.030060120240481, 4.044088176352705, 4.05811623246493, 4.072144288577155, 4.086172344689379, 4.100200400801603, 4.114228456913828, 4.128256513026052, 4.142284569138276, 4.156312625250501, 4.170340681362726, 4.18436873747495, 4.198396793587174, 4.212424849699399, 4.226452905811623, 4.240480961923848, 4.254509018036073, 4.268537074148297, 4.2825651302605205, 4.296593186372745, 4.31062124248497, 4.324649298597194, 4.338677354709419, 4.352705410821644, 4.3667334669338675, 4.380761523046092, 4.394789579158317, 4.408817635270541, 4.422845691382765, 4.436873747494991, 4.4509018036072145, 4.4649298597194385, 4.478957915831663, 4.492985971943888, 4.507014028056112, 4.521042084168337, 4.5350701402805615, 4.5490981963927855, 4.56312625250501, 4.577154308617235, 4.591182364729459, 4.605210420841683, 4.619238476953908, 4.6332665330661325, 4.647294589178356, 4.661322645290581, 4.675350701402806, 4.68937875751503, 4.703406813627255, 4.7174348697394795, 4.731462925851703, 4.745490981963927, 4.759519038076153, 4.773547094188377, 4.787575150300601, 4.801603206412826, 4.81563126252505, 4.829659318637274, 4.843687374749499, 4.857715430861724, 4.871743486973948, 4.885771543086173, 4.899799599198397, 4.913827655310621, 4.927855711422845, 4.94188376753507, 4.955911823647295, 4.969939879759519, 4.9839679358717435, 4.997995991983968, 5.012024048096192, 5.026052104208417, 5.040080160320642, 5.054108216432866, 5.0681362725450905, 5.082164328657314, 5.096192384769539, 5.110220440881764, 5.124248496993988, 5.138276553106213, 5.152304609218437, 5.166332665330661, 5.180360721442886, 5.19438877755511, 5.208416833667335, 5.222444889779559, 5.236472945891784, 5.250501002004008, 5.264529058116232, 5.278557114228457, 5.292585170340681, 5.306613226452906, 5.320641282565131, 5.3346693386773545, 5.348697394789579, 5.362725450901804, 5.376753507014028, 5.390781563126253, 5.404809619238477, 5.4188376753507015, 5.432865731462926, 5.44689378757515, 5.460921843687375, 5.474949899799599, 5.488977955911824, 5.5030060120240485, 5.517034068136272, 5.531062124248497, 5.545090180360721, 5.559118236472946, 5.573146292585171, 5.587174348697395, 5.601202404809619, 5.615230460921843, 5.629258517034068, 5.643286573146293, 5.657314629258517, 5.671342685370742, 5.685370741482966, 5.69939879759519, 5.713426853707415, 5.727454909819639, 5.741482965931864, 5.755511022044089, 5.7695390781563125, 5.783567134268537, 5.797595190380761, 5.811623246492986, 5.825651302605211, 5.839679358717435, 5.8537074148296595, 5.867735470941883, 5.881763527054108, 5.895791583166333, 5.909819639278557, 5.923847695390782, 5.937875751503006, 5.95190380761523, 5.965931863727455, 5.979959919839679, 5.993987975951904, 6.008016032064129, 6.022044088176353, 6.036072144288577, 6.050100200400801, 6.064128256513026, 6.078156312625251, 6.092184368737475, 6.1062124248497, 6.1202404809619235, 6.134268537074148, 6.148296593186373, 6.162324649298597, 6.176352705410822, 6.190380761523046, 6.2044088176352705, 6.218436873747495, 6.232464929859719, 6.246492985971944, 6.260521042084169, 6.274549098196393, 6.2885771543086175, 6.302605210420841, 6.316633266533066, 6.330661322645291, 6.344689378757515, 6.35871743486974, 6.372745490981964, 6.386773547094188, 6.400801603206413, 6.414829659318637, 6.428857715430862, 6.442885771543086, 6.456913827655311, 6.470941883767535, 6.484969939879759, 6.498997995991984, 6.513026052104208, 6.527054108216433, 6.541082164328658, 6.5551102204408815, 6.569138276553106, 6.583166332665331, 6.597194388777555, 6.61122244488978, 6.625250501002004, 6.6392785571142285, 6.653306613226453, 6.667334669338677, 6.681362725450902, 6.695390781563126, 6.709418837675351, 6.7234468937875755, 6.7374749498997994, 6.751503006012024, 6.765531062124248, 6.779559118236473, 6.793587174348698, 6.807615230460922, 6.8216432865731464, 6.835671342685371, 6.849699398797595, 6.86372745490982, 6.877755511022044, 6.891783567134269, 6.9058116232464934, 6.919839679358717, 6.933867735470942, 6.947895791583166, 6.961923847695391, 6.975951903807616, 6.98997995991984, 7.004008016032064, 7.018036072144288, 7.032064128256513, 7.046092184368738, 7.060120240480962, 7.074148296593187, 7.0881763527054105, 7.102204408817635, 7.11623246492986, 7.130260521042084, 7.144288577154309, 7.158316633266534, 7.1723446893787575, 7.186372745490982, 7.200400801603206, 7.214428857715431, 7.228456913827656, 7.24248496993988, 7.2565130260521045, 7.270541082164328, 7.284569138276553, 7.298597194388778, 7.312625250501002, 7.326653306613227, 7.340681362725451, 7.354709418837675, 7.3687374749499, 7.382765531062124, 7.396793587174349, 7.410821643286573, 7.424849699398798, 7.438877755511022, 7.452905811623246, 7.466933867735471, 7.480961923847696, 7.49498997995992, 7.509018036072145, 7.5230460921843685, 7.537074148296593, 7.551102204408818, 7.565130260521042, 7.579158316633267, 7.593186372745491, 7.6072144288577155, 7.62124248496994, 7.635270541082164, 7.649298597194389, 7.663326653306613, 7.677354709418838, 7.6913827655310625, 7.705410821643286, 7.719438877755511, 7.733466933867735, 7.74749498997996, 7.761523046092185, 7.775551102204409, 7.789579158316633, 7.803607214428858, 7.817635270541082, 7.831663326653307, 7.845691382765531, 7.859719438877756, 7.87374749498998, 7.887775551102204, 7.901803607214429, 7.915831663326653, 7.929859719438878, 7.943887775551103, 7.9579158316633265, 7.971943887775551, 7.985971943887775, 8], 
    'y'    : [-3.3056925000000055, -3.2079856682906933, -3.1306044112085, -3.072795233623297, -3.033804640404948, -3.012879136423321, -3.0092652265482838, -3.0222094156497024, -3.0509582085974447, -3.0947581102613766, -3.152855625511366, -3.2244972592172805, -3.308929516248986, -3.405398901476351, -3.5131519197692396, -3.6314350759975205, -3.7594948750310646, -3.8965778217397347, -4.0419304209933955, -4.194799177661918, -4.3544305966151695, -4.520071182723016, -4.690967440855324, -4.866365875881961, -5.045512992672792, -5.227655296097691, -5.4120392910265185, -5.597928728936575, -5.7847442463518615, -5.971990223891665, -6.159171827826106, -6.345794224425319, -6.531362579959427, -6.715382060698564, -6.897357832912857, -7.076795062872435, -7.253198916847428, -7.426074561107957, -7.594927161924161, -7.759261885566164, -7.918583898304099, -8.072398366408084, -8.220210456148257, -8.361525333794747, -8.495848165617678, -8.622684117887184, -8.741538356873384, -8.851916048846421, -8.953322360076411, -9.045262456833493, -9.127241505387786, -9.198764672009425, -9.259337122968539, -9.308480057794108, -9.346133344212058, -9.372685275345846, -9.388546137815299, -9.394126218240245, -9.389835803240512, -9.37608517943593, -9.353284633446325, -9.321844451891526, -9.282174921391361, -9.234686328565655, -9.179788960034243, -9.117893102416945, -9.049409042333593, -8.974747066404017, -8.894317461248042, -8.808530513485499, -8.717796509736214, -8.622525736620013, -8.523128480756725, -8.420015028766183, -8.31359566726821, -8.204280682882633, -8.092480362229288, -7.978604991927992, -7.863064858598582, -7.746269829919136, -7.628551804581346, -7.5100737692019575, -7.390976617766593, -7.271401244260895, -7.151488542670497, -7.031379406981025, -6.911214731178127, -6.7911354092474285, -6.671282335174565, -6.551796402945173, -6.432818506544881, -6.3144895399593315, -6.196950397174154, -6.08034197217498, -5.964805158947456, -5.850480851477202, -5.737509943749859, -5.6260333297510625, -5.51619190346644, -5.408126558881638, -5.301978189982276, -5.197887690754, -5.095995955182442, -4.9964438772532285, -4.899372350952003, -4.804922270264398, -4.713222219267962, -4.624343105730497, -4.538338959843495, -4.455263809687703, -4.37517168334386, -4.298116608892702, -4.224152614414977, -4.153333727991418, -4.08571397770277, -4.0213473916297735, -3.9602879978531664, -3.9025898244536967, -3.8483068995120964, -3.7974932511091106, -3.7502029073254803, -3.7064898962419432, -3.666408245939245, -3.6300119844981205, -3.5973551399993156, -3.5684917405235677, -3.543475814151618, -3.52236138896421, -3.50520249304208, -3.4920531544659714, -3.4829674013166243, -3.4779992616747797, -3.477205446343153, -3.4806719477234607, -3.4885027826412127, -3.5008022316761105, -3.5176745754078556, -3.5392240944161464, -3.565555069280683, -3.596771780581169, -3.6329785088973, -3.67427953480878, -3.720779138895309, -3.7725816017365847, -3.8297912039123103, -3.892512226002183, -3.9608489485859044, -4.034905652243175, -4.114786617553697, -4.20059612509717, -4.292438455453291, -4.390417889201761, -4.494638706922283, -4.605205189194555, -4.722221616598282, -4.845792269713156, -4.976021429118883, -5.113013375395162, -5.256863372953677, -5.407356139090828, -5.563894683177266, -5.7258585760687355, -5.892627388620974, -6.0635806916897295, -6.238098056130732, -6.415559052799726, -6.595343252552452, -6.776830226244658, -6.959399544732074, -7.14243077887044, -7.325303499515506, -7.507397277523004, -7.68809168374869, -7.866766289048283, -8.042800664277536, -8.215574380292185, -8.384467007947977, -8.54885811810065, -8.708127281605941, -8.86165406931959, -9.008818052097343, -9.148998800794944, -9.281575886268122, -9.405928879372624, -9.521437843461287, -9.627675098160893, -9.724696175712385, -9.812631321990276, -9.891610782869105, -9.961764804223398, -10.02322363192768, -10.076117511856486, -10.120576689884338, -10.156731411885765, -10.184711923735298, -10.204648471307467, -10.216671300476795, -10.220910657117816, -10.21749678710505, -10.206559936313038, -10.188230350616296, -10.162638275889359, -10.129913958006753, -10.090187642843008, -10.043589576272646, -9.990250004170205, -9.93029917241021, -9.863867326867187, -9.791084713415664, -9.712081577930173, -9.626988166285239, -9.535955280057594, -9.439246703967129, -9.337164836325348, -9.2300121089153, -9.118090953520005, -9.00170380192251, -8.881153085905863, -8.756741237253083, -8.628770687747215, -8.497543869171306, -8.363363213308388, -8.226531151941478, -8.08735011685365, -7.946122539827923, -7.803150852647327, -7.658737487094908, -7.513184874953713, -7.366795448006762, -7.219871638037101, -7.072715876827777, -6.92563059616181, -6.7789182278222455, -6.632881203592131, -6.487821955254497, -6.344042914592369, -6.201846513388794, -6.061537452443702, -5.923450474708116, -5.7879416004186846, -5.655367300201394, -5.526084044682212, -5.400448304487127, -5.278816550242124, -5.1615452525731795, -5.048990882106259, -4.941509909467371, -4.839458805282483, -4.74319404017757, -4.653072084778619, -4.5694494097116145, -4.492682485602529, -4.423127783077348, -4.361141772762056, -4.307080925282626, -4.261301711265044, -4.2241606013352895, -4.196014066119343, -4.177218576243185, -4.168130602332797, -4.16910661501416, -4.180503084913256, -4.202676482656062, -4.235972484881853, -4.280226838898651, -4.334555304778626, -4.398019263558869, -4.469680096276472, -4.548599183968542, -4.63383790767215, -4.724457648424398, -4.819519787262389, -4.9180857052232065, -5.019216783343941, -5.121974402661693, -5.225419944213552, -5.328614789036605, -5.430620318167959, -5.530497912644696, -5.627308953503907, -5.720114821782692, -5.807976898518145, -5.889956564747357, -5.965115201507415, -6.032514189835423, -6.091214910768465, -6.140278745343636, -6.178767074598032, -6.205741279568743, -6.220263075787817, -6.221780060528889, -6.21086891878364, -6.1883117572575195, -6.1548906826559815, -6.111387801684484, -6.05858522104847, -5.997265047453405, -5.928209387604728, -5.8522003482079015, -5.77002003596838, -5.682450557591607, -5.590274019783048, -5.494272529248143, -5.395228192692351, -5.293923116821132, -5.191139408339923, -5.087659173954195, -4.984264520369386, -4.881737554290956, -4.780860382424361, -4.682415111475045, -4.587183848148466, -4.495948699150082, -4.409491771185336, -4.328595170959693, -4.254041005178592, -4.1865055414167855, -4.125980529570414, -4.07218536981938, -4.024838764673548, -3.9836594166427726, -3.948366028236917, -3.91867730196584, -3.8943119403393984, -3.8749886458674534, -3.8604261210598643, -3.8503430684264894, -3.844458190477189, -3.8424901897218215, -3.8441577686702475, -3.8491796298323235, -3.8572744757179134, -3.868161008836872, -3.881557931699061, -3.8971839468143394, -3.9147577566925644, -3.9339980638435983, -3.9546235707772976, -3.9763529800035253, -3.998904994032136, -4.021998315372992, -4.045351646535952, -4.0686895605255975, -4.091832234286118, -4.11467762550884, -4.137125950106946, -4.15907742399363, -4.180432263082075, -4.2010906832854715, -4.2209529005170054, -4.239919130689864, -4.2578895897172355, -4.274764493512308, -4.290444057988269, -4.304828499058305, -4.317818032635602, -4.3293128746333505, -4.339213240964739, -4.34741934754295, -4.3538314102811775, -4.358349645092603, -4.360874267890419, -4.36130549458781, -4.359543541097963, -4.355488623334068, -4.349040957209311, -4.34010075863688, -4.328568243529963, -4.314344778594985, -4.297410373404096, -4.277872796889926, -4.255851523044105, -4.231466025858269, -4.204835779324055, -4.176080257433095, -4.145318934177023, -4.112671283547479, -4.0782567795360904, -4.042194896134496, -4.004605107334327, -3.9656068871272203, -3.925319709504811, -3.8838630484587315, -3.841356377980619, -3.7979191720621026, -3.753670904694823, -3.7087310498704142, -3.6632190815805044, -3.6172544738167356, -3.5709567005707368, -3.5244452358341447, -3.4778395535985966, -3.431259127855721, -3.3848234325971593, -3.3386519435748365, -3.292878211885129, -3.247683888772836, -3.2032608915418734, -3.159801137496168, -3.1174965439396436, -3.076539028176218, -3.037120507509818, -2.9994328992443617, -2.963668120683774, -2.9300180891319787, -2.8986747218928945, -2.8698299362704476, -2.843675649568558, -2.820403779091148, -2.8002062421421416, -2.7832749560254606, -2.7698018380450264, -2.7599788055047623, -2.7539977757085903, -2.7520506659604336, -2.754329393564214, -2.761025875823854, -2.772332030043276, -2.788439773526402, -2.8095410235771543, -2.835827697499457, -2.867462529066319, -2.9043851255821065, -2.9464321002662435, -2.993439512931929, -3.045243423392374, -3.1016798914607806, -3.162584976950348, -3.2277947396742888, -3.2971452394457996, -3.370472536078093, -3.44761268938437, -3.5284017591778287, -3.6126758052716843, -3.700270887479131, -3.791023065613383, -3.884768399487642, -3.9813429489151035, -4.080582773708986, -4.1823239336824845, -4.286402488648801, -4.392654498421152, -4.500916022812726, -4.6110231216367445, -4.722811854706403, -4.8361182818348984, -4.950778462835451, -5.066628068413683, -5.18349483576105, -5.301199095017329, -5.419560892662878, -5.538400275178085, -5.657537289043297, -5.7767919807389, -5.895984396745255, -6.014934583542717, -6.133462587611674, -6.251388455432476, -6.368532233485507, -6.484713968251126, -6.599753706209693, -6.713471493841595, -6.825687377627188, -6.93622140404683, -7.0448936195809075, -7.151524070709773, -7.255932803913808, -7.357939865673372, -7.457365302468828, -7.554029160780558, -7.647751487088912, -7.738352327874273, -7.825651729617001, -7.909469806749401, -7.989633916836864, -8.065984986631541, -8.13836543456657, -8.20661767907507, -8.27058413859017, -8.330107231545, -8.385029376372682, -8.435192991506351, -8.480440495379126, -8.520614306424136, -8.555556843074513, -8.585110523763378, -8.609117766923855, -8.62742099098908, -8.639862614392174, -8.646285055566269, -8.646530732944484, -8.640442064959952, -8.6278614700458, -8.608631366635155, -8.58259417316114, -8.549592308056884, -8.509468189755516, -8.46206423669016, -8.40722286729395, -8.3447865]
    }
    trace2 = {
    'fill': 'tonexty',
    'line': {
        'color': '#6a3f39',
        'shape': 'spline',
        'width': 0
    },
    'mode': 'lines',
    'name': '',
    'type': 'scatter',
    'x': [1, 1.0140280561122244, 1.028056112224449, 1.0420841683366733, 1.0561122244488979, 1.0701402805611222, 1.0841683366733468, 1.0981963927855711, 1.1122244488977957, 1.12625250501002, 1.1402805611222444, 1.154308617234469, 1.1683366733466933, 1.182364729458918, 1.1963927855711423, 1.2104208416833666, 1.2244488977955912, 1.2384769539078158, 1.25250501002004, 1.2665330661322645, 1.280561122244489, 1.2945891783567134, 1.308617234468938, 1.3226452905811623, 1.3366733466933867, 1.3507014028056112, 1.3647294589178358, 1.3787575150300602, 1.3927855711422845, 1.406813627254509, 1.4208416833667334, 1.434869739478958, 1.4488977955911824, 1.4629258517034067, 1.4769539078156313, 1.4909819639278556, 1.5050100200400802, 1.5190380761523046, 1.533066132264529, 1.5470941883767535, 1.561122244488978, 1.5751503006012024, 1.5891783567134268, 1.6032064128256514, 1.6172344689378757, 1.6312625250501003, 1.6452905811623246, 1.659318637274549, 1.6733466933867736, 1.6873747494989981, 1.7014028056112225, 1.7154308617234468, 1.7294589178356714, 1.7434869739478958, 1.7575150300601203, 1.7715430861723447, 1.785571142284569, 1.7995991983967936, 1.8136272545090182, 1.8276553106212425, 1.8416833667334669, 1.8557114228456912, 1.8697394789579158, 1.8837675350701404, 1.8977955911823647, 1.911823647294589, 1.9258517034068137, 1.9398797595190382, 1.9539078156312626, 1.967935871743487, 1.9819639278557113, 1.9959919839679359, 2.0100200400801604, 2.024048096192385, 2.038076152304609, 2.052104208416834, 2.066132264529058, 2.0801603206412826, 2.094188376753507, 2.1082164328657313, 2.122244488977956, 2.13627254509018, 2.150300601202405, 2.164328657314629, 2.1783567134268536, 2.1923847695390783, 2.2064128256513027, 2.220440881763527, 2.2344689378757514, 2.2484969939879758, 2.2625250501002006, 2.276553106212425, 2.2905811623246493, 2.304609218436874, 2.318637274549098, 2.3326653306613228, 2.346693386773547, 2.3607214428857715, 2.3747494989979963, 2.38877755511022, 2.402805611222445, 2.4168336673346693, 2.4308617234468937, 2.4448897795591185, 2.458917835671343, 2.472945891783567, 2.4869739478957915, 2.501002004008016, 2.5150300601202407, 2.529058116232465, 2.5430861723446894, 2.557114228456914, 2.571142284569138, 2.585170340681363, 2.599198396793587, 2.6132264529058116, 2.6272545090180364, 2.6412825651302603, 2.655310621242485, 2.6693386773547094, 2.6833667334669338, 2.6973947895791586, 2.7114228456913825, 2.7254509018036073, 2.7394789579158316, 2.753507014028056, 2.7675350701402808, 2.781563126252505, 2.7955911823647295, 2.809619238476954, 2.823647294589178, 2.837675350701403, 2.8517034068136273, 2.8657314629258517, 2.8797595190380765, 2.8937875751503004, 2.907815631262525, 2.9218436873747495, 2.935871743486974, 2.9498997995991987, 2.9639278557114226, 2.9779559118236474, 2.9919839679358717, 3.006012024048096, 3.020040080160321, 3.0340681362725452, 3.0480961923847696, 3.062124248496994, 3.0761523046092183, 3.090180360721443, 3.1042084168336674, 3.118236472945892, 3.132264529058116, 3.1462925851703405, 3.1603206412825653, 3.1743486973947896, 3.188376753507014, 3.2024048096192383, 3.216432865731463, 3.2304609218436875, 3.244488977955912, 3.258517034068136, 3.2725450901803605, 3.2865731462925853, 3.3006012024048097, 3.314629258517034, 3.3286573146292584, 3.342685370741483, 3.3567134268537075, 3.370741482965932, 3.3847695390781563, 3.3987975951903806, 3.4128256513026054, 3.4268537074148298, 3.440881763527054, 3.4549098196392785, 3.468937875751503, 3.4829659318637276, 3.496993987975952, 3.5110220440881763, 3.5250501002004007, 3.5390781563126255, 3.55310621242485, 3.567134268537074, 3.5811623246492985, 3.595190380761523, 3.6092184368737477, 3.623246492985972, 3.6372745490981964, 3.6513026052104207, 3.6653306613226455, 3.67935871743487, 3.693386773547094, 3.7074148296593186, 3.721442885771543, 3.7354709418837677, 3.749498997995992, 3.7635270541082164, 3.7775551102204408, 3.7915831663326656, 3.80561122244489, 3.8196392785571143, 3.8336673346693386, 3.847695390781563, 3.8617234468937878, 3.875751503006012, 3.8897795591182365, 3.903807615230461, 3.9178356713426856, 3.93186372745491, 3.9458917835671343, 3.9599198396793587, 3.973947895791583, 3.987975951903808, 4.002004008016032, 4.0160320641282565, 4.030060120240481, 4.044088176352705, 4.05811623246493, 4.072144288577155, 4.086172344689379, 4.100200400801603, 4.114228456913828, 4.128256513026052, 4.142284569138276, 4.156312625250501, 4.170340681362726, 4.18436873747495, 4.198396793587174, 4.212424849699399, 4.226452905811623, 4.240480961923848, 4.254509018036073, 4.268537074148297, 4.2825651302605205, 4.296593186372745, 4.31062124248497, 4.324649298597194, 4.338677354709419, 4.352705410821644, 4.3667334669338675, 4.380761523046092, 4.394789579158317, 4.408817635270541, 4.422845691382765, 4.436873747494991, 4.4509018036072145, 4.4649298597194385, 4.478957915831663, 4.492985971943888, 4.507014028056112, 4.521042084168337, 4.5350701402805615, 4.5490981963927855, 4.56312625250501, 4.577154308617235, 4.591182364729459, 4.605210420841683, 4.619238476953908, 4.6332665330661325, 4.647294589178356, 4.661322645290581, 4.675350701402806, 4.68937875751503, 4.703406813627255, 4.7174348697394795, 4.731462925851703, 4.745490981963927, 4.759519038076153, 4.773547094188377, 4.787575150300601, 4.801603206412826, 4.81563126252505, 4.829659318637274, 4.843687374749499, 4.857715430861724, 4.871743486973948, 4.885771543086173, 4.899799599198397, 4.913827655310621, 4.927855711422845, 4.94188376753507, 4.955911823647295, 4.969939879759519, 4.9839679358717435, 4.997995991983968, 5.012024048096192, 5.026052104208417, 5.040080160320642, 5.054108216432866, 5.0681362725450905, 5.082164328657314, 5.096192384769539, 5.110220440881764, 5.124248496993988, 5.138276553106213, 5.152304609218437, 5.166332665330661, 5.180360721442886, 5.19438877755511, 5.208416833667335, 5.222444889779559, 5.236472945891784, 5.250501002004008, 5.264529058116232, 5.278557114228457, 5.292585170340681, 5.306613226452906, 5.320641282565131, 5.3346693386773545, 5.348697394789579, 5.362725450901804, 5.376753507014028, 5.390781563126253, 5.404809619238477, 5.4188376753507015, 5.432865731462926, 5.44689378757515, 5.460921843687375, 5.474949899799599, 5.488977955911824, 5.5030060120240485, 5.517034068136272, 5.531062124248497, 5.545090180360721, 5.559118236472946, 5.573146292585171, 5.587174348697395, 5.601202404809619, 5.615230460921843, 5.629258517034068, 5.643286573146293, 5.657314629258517, 5.671342685370742, 5.685370741482966, 5.69939879759519, 5.713426853707415, 5.727454909819639, 5.741482965931864, 5.755511022044089, 5.7695390781563125, 5.783567134268537, 5.797595190380761, 5.811623246492986, 5.825651302605211, 5.839679358717435, 5.8537074148296595, 5.867735470941883, 5.881763527054108, 5.895791583166333, 5.909819639278557, 5.923847695390782, 5.937875751503006, 5.95190380761523, 5.965931863727455, 5.979959919839679, 5.993987975951904, 6.008016032064129, 6.022044088176353, 6.036072144288577, 6.050100200400801, 6.064128256513026, 6.078156312625251, 6.092184368737475, 6.1062124248497, 6.1202404809619235, 6.134268537074148, 6.148296593186373, 6.162324649298597, 6.176352705410822, 6.190380761523046, 6.2044088176352705, 6.218436873747495, 6.232464929859719, 6.246492985971944, 6.260521042084169, 6.274549098196393, 6.2885771543086175, 6.302605210420841, 6.316633266533066, 6.330661322645291, 6.344689378757515, 6.35871743486974, 6.372745490981964, 6.386773547094188, 6.400801603206413, 6.414829659318637, 6.428857715430862, 6.442885771543086, 6.456913827655311, 6.470941883767535, 6.484969939879759, 6.498997995991984, 6.513026052104208, 6.527054108216433, 6.541082164328658, 6.5551102204408815, 6.569138276553106, 6.583166332665331, 6.597194388777555, 6.61122244488978, 6.625250501002004, 6.6392785571142285, 6.653306613226453, 6.667334669338677, 6.681362725450902, 6.695390781563126, 6.709418837675351, 6.7234468937875755, 6.7374749498997994, 6.751503006012024, 6.765531062124248, 6.779559118236473, 6.793587174348698, 6.807615230460922, 6.8216432865731464, 6.835671342685371, 6.849699398797595, 6.86372745490982, 6.877755511022044, 6.891783567134269, 6.9058116232464934, 6.919839679358717, 6.933867735470942, 6.947895791583166, 6.961923847695391, 6.975951903807616, 6.98997995991984, 7.004008016032064, 7.018036072144288, 7.032064128256513, 7.046092184368738, 7.060120240480962, 7.074148296593187, 7.0881763527054105, 7.102204408817635, 7.11623246492986, 7.130260521042084, 7.144288577154309, 7.158316633266534, 7.1723446893787575, 7.186372745490982, 7.200400801603206, 7.214428857715431, 7.228456913827656, 7.24248496993988, 7.2565130260521045, 7.270541082164328, 7.284569138276553, 7.298597194388778, 7.312625250501002, 7.326653306613227, 7.340681362725451, 7.354709418837675, 7.3687374749499, 7.382765531062124, 7.396793587174349, 7.410821643286573, 7.424849699398798, 7.438877755511022, 7.452905811623246, 7.466933867735471, 7.480961923847696, 7.49498997995992, 7.509018036072145, 7.5230460921843685, 7.537074148296593, 7.551102204408818, 7.565130260521042, 7.579158316633267, 7.593186372745491, 7.6072144288577155, 7.62124248496994, 7.635270541082164, 7.649298597194389, 7.663326653306613, 7.677354709418838, 7.6913827655310625, 7.705410821643286, 7.719438877755511, 7.733466933867735, 7.74749498997996, 7.761523046092185, 7.775551102204409, 7.789579158316633, 7.803607214428858, 7.817635270541082, 7.831663326653307, 7.845691382765531, 7.859719438877756, 7.87374749498998, 7.887775551102204, 7.901803607214429, 7.915831663326653, 7.929859719438878, 7.943887775551103, 7.9579158316633265, 7.971943887775551, 7.985971943887775, 8], 
    'y': [-2.3056915000000036, -2.208232138997963, -2.1310657936697774, -2.0734405599455226, -2.0346045337552674, -2.0138058110290844, -2.0102924876970465, -2.023312659689225, -2.0521144229356922, -2.095945873366519, -2.1540551069117786, -2.2256902195015433, -2.310099307065884, -2.406530465534874, -2.5142317908385827, -2.6324513789070836, -2.760437325670452, -2.897437727058756, -3.0427006790020656, -3.1954742774304563, -3.355006618274001, -3.520545797462769, -3.6913399109268337, -3.866637054596266, -4.045685324401138, -4.227732816271525, -4.412027626137496, -4.597834959235932, -4.784575656885112, -4.971753637810011, -5.158873600131854, -5.3454402419718825, -5.530958261451325, -5.714932356691421, -5.896867225813404, -6.076267566938507, -6.252638078187967, -6.425483457683011, -6.594308403544884, -6.758617613894813, -6.917915786854039, -7.071707620543786, -7.2194978130852965, -7.360791062599806, -7.495092067208545, -7.62190552503275, -7.74073613419365, -7.85108859281249, -7.952467599010494, -8.044377850908905, -8.12632404662895, -8.197810884291867, -8.258343062018893, -8.307441246149635, -8.345043072852237, -8.371532784610991, -8.387316528187787, -8.392800450344522, -8.388390697843086, -8.374493417445379, -8.351514755913286, -8.319860860008706, -8.279937876493536, -8.232151952129662, -8.176909233678984, -8.114615867903392, -8.04567800156478, -7.970501781425048, -7.889493354246081, -7.80305886678978, -7.711604465818035, -7.615536298092739, -7.515260510375785, -7.411183249429072, -7.30371066201449, -7.193248894893932, -7.080204094829299, -6.964982408582472, -6.847989982915356, -6.729632530949851, -6.610235061340387, -6.489947744518035, -6.368897883180006, -6.247212780023535, -6.125019737745844, -6.002446059044152, -5.879619046615696, -5.756666003157693, -5.633714231367367, -5.510891033941948, -5.388323713578654, -5.2661395729747165, -5.144465914827357, -5.023430041833798, -4.903159256691277, -4.783780862097001, -4.665422160748205, -4.548210455342115, -4.432273048575949, -4.317737243146942, -4.204730341752306, -4.093379647089277, -3.9838124618550768, -3.8761560887469235, -3.770537830462051, -3.667084989697683, -3.5659108402668647, -3.4670629256568244, -3.37056955496558, -3.2764590348856535, -3.1847596721095597, -3.0954997733298124, -3.0087076452389363, -2.9244115945294373, -2.8426399278938397, -2.763420952024659, -2.6867829736144095, -2.6127542993556157, -2.541363235940784, -2.4726380900624374, -2.4066071684130934, -2.3432987776852636, -2.282741224571473, -2.22496281576423, -2.1699918579560578, -2.11785665783947, -2.068585522106984, -2.022206757451118, -1.978748670564387, -1.9382395681393096, -1.9007077568684014, -1.8661815434441789, -1.8346992870463488, -1.8064090685855165, -1.781526508691623, -1.7602682163137942, -1.7428508004011494, -1.7294908699028122, -1.7204050337679044, -1.7158099009455485, -1.7159220803848667, -1.7209581810349805, -1.7311348118450136, -1.7466685817640868, -1.7677760997413223, -1.7946739747258427, -1.82757881566677, -1.8667072315132258, -1.9122758312143344, -1.9645012237192172, -2.023600017976994, -2.0897888229367885, -2.163284247547723, -2.24430290075892, -2.333061391519504, -2.429776328778592, -2.534664321485309, -2.6479419785887766, -2.7698129641138975, -2.900035075408476, -3.0378180743174803, -3.1823380709395135, -3.332771175373173, -3.488293497717065, -3.6480811480697803, -3.8113102365299225, -3.977156873196089, -4.144797168166889, -4.3134072315409115, -4.482163173416756, -4.6502411038930305, -4.816817133068328, -4.981067371041259, -5.142167927910409, -5.299294913774384, -5.451624438731784, -5.59833261288121, -5.738595546321263, -5.871589349150539, -5.996490131467635, -6.112474003371158, -6.21871707495971, -6.314395456331881, -6.398685257586275, -6.470763532723442, -6.530175807096828, -6.577393713707879, -6.613032078979412, -6.637705729334268, -6.652029491195275, -6.656618190985258, -6.652086655127056, -6.639049710043494, -6.618122182157403, -6.5899188978916134, -6.555054683668958, -6.514144365912267, -6.467802771044371, -6.416644725488094, -6.361285055666277, -6.302338588001744, -6.240420148917327, -6.176144564835855, -6.110126662180161, -6.04298126737307, -5.97532320683742, -5.907767306996039, -5.8409283942717565, -5.7754212950874, -5.7118608358658065, -5.650861843029801, -5.592932138606653, -5.537991401595374, -5.485758304773386, -5.435951346679335, -5.388289025851845, -5.342489840829561, -5.298272290151124, -5.255354872355161, -5.213456085980313, -5.172294429565222, -5.13158840164852, -5.09105650076884, -5.0504172254648285, -5.009389074275116, -4.967690545738342, -4.92504013839314, -4.881156350778152, -4.835757681432012, -4.788562628893356, -4.739289691700826, -4.687657368393052, -4.633384157508674, -4.576188557586335, -4.515789067164666, -4.4519041847822995, -4.384252408977877, -4.3125822492055175, -4.237039564303063, -4.158051634910655, -4.076051698696212, -3.9914729933276387, -3.9047487564728574, -3.816312225799785, -3.7265966389763334, -3.6360352336704054, -3.545061247549938, -3.454107918282835, -3.363608483537006, -3.2739961809803724, -3.185704248280852, -3.0991659231063506, -3.0148144431247883, -2.9330830460040853, -2.854404969412145, -2.7792134510168895, -2.7079417284862353, -2.6410230394880942, -2.5788906216903773, -2.5219777127610037, -2.4707175503678913, -2.425543372178949, -2.386888415862093, -2.3551787183446837, -2.330500139294548, -2.3124582300901726, -2.300622265492882, -2.2945615202639957, -2.2938452691648403, -2.2980427869567377, -2.30672334840101, -2.3194562282589852, -2.335810701291983, -2.3553560422613287, -2.377661525928344, -2.402296427054352, -2.428830020400677, -2.456831580728646, -2.485870382799576, -2.515515701374793, -2.545336811215622, -2.5749029870833864, -2.6037835037394084, -2.6315476359450103, -2.6577646584615184, -2.682003846050253, -2.70383447347254, -2.7228258154897014, -2.7385471468630618, -2.7505678795710264, -2.7586157226498433, -2.7628815614007802, -2.763640549566449, -2.7611678408894638, -2.7557385891124384, -2.747627947977982, -2.737111071228713, -2.7244631126072365, -2.709959225856173, -2.6938745647181324, -2.6764842829357245, -2.658063534251568, -2.63888747240827, -2.6192312511484466, -2.5993700242147106, -2.5795789453496716, -2.5601331682959474, -2.541307846796147, -2.523378134592885, -2.506619185428774, -2.491306153046425, -2.477714191188453, -2.46611845359747, -2.456794094016089, -2.450016266186924, -2.4460601238525843, -2.4451586581396816, -2.447272172220581, -2.4522524746454097, -2.459951096036896, -2.4702195670177667, -2.4829094182107507, -2.4978721802385726, -2.514959383723963, -2.5340225592896455, -2.5549132375583525, -2.5774829491528086, -2.60158322469574, -2.6270655948098773, -2.653781590117945, -2.681582741242673, -2.7103205788067886, -2.7398466334330167, -2.770012435744088, -2.8006695163627287, -2.831669405911663, -2.862863635013624, -2.8941037342913347, -2.9252412343675265, -2.9561276658649227, -2.9866145594062523, -3.016553445614245, -3.0457983225263154, -3.0742433712608923, -3.101815464790696, -3.12844242523669, -3.154052074719849, -3.1785722353611376, -3.2019307292815276, -3.2240553786019865, -3.2448740054434824, -3.2643144319269854, -3.2823044801734635, -3.298771972303888, -3.313644730439224, -3.32685057670044, -3.3383173332085088, -3.3479728220843983, -3.355744865449074, -3.3615612854235093, -3.3653499041286685, -3.367038543685525, -3.366555026215044, -3.3638271738381955, -3.3587828086759486, -3.3513497528492726, -3.3414558284791345, -3.3290288576865055, -3.3139976393123414, -3.296357719251994, -3.2762130774069407, -3.2536776281870528, -3.2288652860022076, -3.2018899652622848, -3.1728655803771546, -3.1419060457566945, -3.109125275810784, -3.074637184949294, -3.038555687582103, -3.000994698119084, -2.962068130970115, -2.9218899005450725, -2.8805739212538297, -2.8382341075062656, -2.7949843737122504, -2.7509386342816655, -2.7062108036243866, -2.660914796150284, -2.61516452626924, -2.5690739083911245, -2.522756856925816, -2.476327286283193, -2.4298991108731256, -2.383586245105496, -2.3375026051313057, -2.291776029206569, -2.246581932065427, -2.2021058827337012, -2.1585334502372255, -2.1160502036018296, -2.074841711853338, -2.035093544017585, -1.9969912691203944, -1.9607204561875975, -1.9264666742450252, -1.8944154923185021, -1.8647524794338615, -1.8376632046169292, -1.8133332368935349, -1.791948145289509, -1.7736934988306787, -1.7587548665428734, -1.7473178174519222, -1.7395679205836534, -1.7356907449638972, -1.735871859618482, -1.740296833573236, -1.7491512358539894, -1.7626206354865699, -1.7808906014968067, -1.80414670291053, -1.8325417159175839, -1.8659776948293771, -1.9042409619663885, -1.947117217799755, -1.9943921628006251, -2.045851497440137, -2.1012809221894297, -2.160466137519651, -2.223192843901937, -2.2892467418074345, -2.358413531707285, -2.4304789140726233, -2.505228589374601, -2.5824482580843506, -2.661923620673023, -2.7434403776117575, -2.8267842293716887, -2.9117408764239694, -2.9980960192397332, -3.085635358290122, -3.174144594046285, -3.2634094269793534, -3.353215557560482, -3.4433486862608045, -3.533594513551458, -3.6237387399035965, -3.7135696705941426, -3.8029287202853403, -3.891706888722479, -3.979797074554271, -4.067092176429447, -4.153485092996711, -4.238868722904793, -4.3231359648024075, -4.4061797173382615, -4.487892879161089, -4.5681683489195954, -4.646899025262508, -4.7239778068385405, -4.799297592296405, -4.872751280284834, -4.944231769452532, -5.013631958448217, -5.080844745920615, -5.145763030518436, -5.208279710890406, -5.268287685685237, -5.325679853551645, -5.380349113138358, -5.432188363094079, -5.48109050206754, -5.526948428707449, -5.56965521848298, -5.60912279450914, -5.645298388823739, -5.6781331150272, -5.7075780867199395, -5.733584417502383, -5.756103220974951, -5.77508561073806, -5.790482700392137, -5.802245603537596, -5.810325433774862, -5.814673304704356, -5.815240329926496, -5.811977623041702, -5.804836297650398, -5.793767467353003, -5.778722245749941, -5.759651746441626, -5.736507083028483, -5.709239369110933, -5.677799718289398, -5.6421392441642935, -5.602209060336041, -5.557960280405068, -5.509344017971788, -5.456311386636628, -5.398813499999999], 
    'fillcolor': '#6a3f39'
    }
    trace3 = {
    'fill': 'tonexty',
    'line': {
        'color': '#a87c79',
        'shape': 'spline',
        'width': 0
    },
    'mode': 'lines',
    'name': '',
    'type': 'scatter',
    'x': [1, 1.0140280561122244, 1.028056112224449, 1.0420841683366733, 1.0561122244488979, 1.0701402805611222, 1.0841683366733468, 1.0981963927855711, 1.1122244488977957, 1.12625250501002, 1.1402805611222444, 1.154308617234469, 1.1683366733466933, 1.182364729458918, 1.1963927855711423, 1.2104208416833666, 1.2244488977955912, 1.2384769539078158, 1.25250501002004, 1.2665330661322645, 1.280561122244489, 1.2945891783567134, 1.308617234468938, 1.3226452905811623, 1.3366733466933867, 1.3507014028056112, 1.3647294589178358, 1.3787575150300602, 1.3927855711422845, 1.406813627254509, 1.4208416833667334, 1.434869739478958, 1.4488977955911824, 1.4629258517034067, 1.4769539078156313, 1.4909819639278556, 1.5050100200400802, 1.5190380761523046, 1.533066132264529, 1.5470941883767535, 1.561122244488978, 1.5751503006012024, 1.5891783567134268, 1.6032064128256514, 1.6172344689378757, 1.6312625250501003, 1.6452905811623246, 1.659318637274549, 1.6733466933867736, 1.6873747494989981, 1.7014028056112225, 1.7154308617234468, 1.7294589178356714, 1.7434869739478958, 1.7575150300601203, 1.7715430861723447, 1.785571142284569, 1.7995991983967936, 1.8136272545090182, 1.8276553106212425, 1.8416833667334669, 1.8557114228456912, 1.8697394789579158, 1.8837675350701404, 1.8977955911823647, 1.911823647294589, 1.9258517034068137, 1.9398797595190382, 1.9539078156312626, 1.967935871743487, 1.9819639278557113, 1.9959919839679359, 2.0100200400801604, 2.024048096192385, 2.038076152304609, 2.052104208416834, 2.066132264529058, 2.0801603206412826, 2.094188376753507, 2.1082164328657313, 2.122244488977956, 2.13627254509018, 2.150300601202405, 2.164328657314629, 2.1783567134268536, 2.1923847695390783, 2.2064128256513027, 2.220440881763527, 2.2344689378757514, 2.2484969939879758, 2.2625250501002006, 2.276553106212425, 2.2905811623246493, 2.304609218436874, 2.318637274549098, 2.3326653306613228, 2.346693386773547, 2.3607214428857715, 2.3747494989979963, 2.38877755511022, 2.402805611222445, 2.4168336673346693, 2.4308617234468937, 2.4448897795591185, 2.458917835671343, 2.472945891783567, 2.4869739478957915, 2.501002004008016, 2.5150300601202407, 2.529058116232465, 2.5430861723446894, 2.557114228456914, 2.571142284569138, 2.585170340681363, 2.599198396793587, 2.6132264529058116, 2.6272545090180364, 2.6412825651302603, 2.655310621242485, 2.6693386773547094, 2.6833667334669338, 2.6973947895791586, 2.7114228456913825, 2.7254509018036073, 2.7394789579158316, 2.753507014028056, 2.7675350701402808, 2.781563126252505, 2.7955911823647295, 2.809619238476954, 2.823647294589178, 2.837675350701403, 2.8517034068136273, 2.8657314629258517, 2.8797595190380765, 2.8937875751503004, 2.907815631262525, 2.9218436873747495, 2.935871743486974, 2.9498997995991987, 2.9639278557114226, 2.9779559118236474, 2.9919839679358717, 3.006012024048096, 3.020040080160321, 3.0340681362725452, 3.0480961923847696, 3.062124248496994, 3.0761523046092183, 3.090180360721443, 3.1042084168336674, 3.118236472945892, 3.132264529058116, 3.1462925851703405, 3.1603206412825653, 3.1743486973947896, 3.188376753507014, 3.2024048096192383, 3.216432865731463, 3.2304609218436875, 3.244488977955912, 3.258517034068136, 3.2725450901803605, 3.2865731462925853, 3.3006012024048097, 3.314629258517034, 3.3286573146292584, 3.342685370741483, 3.3567134268537075, 3.370741482965932, 3.3847695390781563, 3.3987975951903806, 3.4128256513026054, 3.4268537074148298, 3.440881763527054, 3.4549098196392785, 3.468937875751503, 3.4829659318637276, 3.496993987975952, 3.5110220440881763, 3.5250501002004007, 3.5390781563126255, 3.55310621242485, 3.567134268537074, 3.5811623246492985, 3.595190380761523, 3.6092184368737477, 3.623246492985972, 3.6372745490981964, 3.6513026052104207, 3.6653306613226455, 3.67935871743487, 3.693386773547094, 3.7074148296593186, 3.721442885771543, 3.7354709418837677, 3.749498997995992, 3.7635270541082164, 3.7775551102204408, 3.7915831663326656, 3.80561122244489, 3.8196392785571143, 3.8336673346693386, 3.847695390781563, 3.8617234468937878, 3.875751503006012, 3.8897795591182365, 3.903807615230461, 3.9178356713426856, 3.93186372745491, 3.9458917835671343, 3.9599198396793587, 3.973947895791583, 3.987975951903808, 4.002004008016032, 4.0160320641282565, 4.030060120240481, 4.044088176352705, 4.05811623246493, 4.072144288577155, 4.086172344689379, 4.100200400801603, 4.114228456913828, 4.128256513026052, 4.142284569138276, 4.156312625250501, 4.170340681362726, 4.18436873747495, 4.198396793587174, 4.212424849699399, 4.226452905811623, 4.240480961923848, 4.254509018036073, 4.268537074148297, 4.2825651302605205, 4.296593186372745, 4.31062124248497, 4.324649298597194, 4.338677354709419, 4.352705410821644, 4.3667334669338675, 4.380761523046092, 4.394789579158317, 4.408817635270541, 4.422845691382765, 4.436873747494991, 4.4509018036072145, 4.4649298597194385, 4.478957915831663, 4.492985971943888, 4.507014028056112, 4.521042084168337, 4.5350701402805615, 4.5490981963927855, 4.56312625250501, 4.577154308617235, 4.591182364729459, 4.605210420841683, 4.619238476953908, 4.6332665330661325, 4.647294589178356, 4.661322645290581, 4.675350701402806, 4.68937875751503, 4.703406813627255, 4.7174348697394795, 4.731462925851703, 4.745490981963927, 4.759519038076153, 4.773547094188377, 4.787575150300601, 4.801603206412826, 4.81563126252505, 4.829659318637274, 4.843687374749499, 4.857715430861724, 4.871743486973948, 4.885771543086173, 4.899799599198397, 4.913827655310621, 4.927855711422845, 4.94188376753507, 4.955911823647295, 4.969939879759519, 4.9839679358717435, 4.997995991983968, 5.012024048096192, 5.026052104208417, 5.040080160320642, 5.054108216432866, 5.0681362725450905, 5.082164328657314, 5.096192384769539, 5.110220440881764, 5.124248496993988, 5.138276553106213, 5.152304609218437, 5.166332665330661, 5.180360721442886, 5.19438877755511, 5.208416833667335, 5.222444889779559, 5.236472945891784, 5.250501002004008, 5.264529058116232, 5.278557114228457, 5.292585170340681, 5.306613226452906, 5.320641282565131, 5.3346693386773545, 5.348697394789579, 5.362725450901804, 5.376753507014028, 5.390781563126253, 5.404809619238477, 5.4188376753507015, 5.432865731462926, 5.44689378757515, 5.460921843687375, 5.474949899799599, 5.488977955911824, 5.5030060120240485, 5.517034068136272, 5.531062124248497, 5.545090180360721, 5.559118236472946, 5.573146292585171, 5.587174348697395, 5.601202404809619, 5.615230460921843, 5.629258517034068, 5.643286573146293, 5.657314629258517, 5.671342685370742, 5.685370741482966, 5.69939879759519, 5.713426853707415, 5.727454909819639, 5.741482965931864, 5.755511022044089, 5.7695390781563125, 5.783567134268537, 5.797595190380761, 5.811623246492986, 5.825651302605211, 5.839679358717435, 5.8537074148296595, 5.867735470941883, 5.881763527054108, 5.895791583166333, 5.909819639278557, 5.923847695390782, 5.937875751503006, 5.95190380761523, 5.965931863727455, 5.979959919839679, 5.993987975951904, 6.008016032064129, 6.022044088176353, 6.036072144288577, 6.050100200400801, 6.064128256513026, 6.078156312625251, 6.092184368737475, 6.1062124248497, 6.1202404809619235, 6.134268537074148, 6.148296593186373, 6.162324649298597, 6.176352705410822, 6.190380761523046, 6.2044088176352705, 6.218436873747495, 6.232464929859719, 6.246492985971944, 6.260521042084169, 6.274549098196393, 6.2885771543086175, 6.302605210420841, 6.316633266533066, 6.330661322645291, 6.344689378757515, 6.35871743486974, 6.372745490981964, 6.386773547094188, 6.400801603206413, 6.414829659318637, 6.428857715430862, 6.442885771543086, 6.456913827655311, 6.470941883767535, 6.484969939879759, 6.498997995991984, 6.513026052104208, 6.527054108216433, 6.541082164328658, 6.5551102204408815, 6.569138276553106, 6.583166332665331, 6.597194388777555, 6.61122244488978, 6.625250501002004, 6.6392785571142285, 6.653306613226453, 6.667334669338677, 6.681362725450902, 6.695390781563126, 6.709418837675351, 6.7234468937875755, 6.7374749498997994, 6.751503006012024, 6.765531062124248, 6.779559118236473, 6.793587174348698, 6.807615230460922, 6.8216432865731464, 6.835671342685371, 6.849699398797595, 6.86372745490982, 6.877755511022044, 6.891783567134269, 6.9058116232464934, 6.919839679358717, 6.933867735470942, 6.947895791583166, 6.961923847695391, 6.975951903807616, 6.98997995991984, 7.004008016032064, 7.018036072144288, 7.032064128256513, 7.046092184368738, 7.060120240480962, 7.074148296593187, 7.0881763527054105, 7.102204408817635, 7.11623246492986, 7.130260521042084, 7.144288577154309, 7.158316633266534, 7.1723446893787575, 7.186372745490982, 7.200400801603206, 7.214428857715431, 7.228456913827656, 7.24248496993988, 7.2565130260521045, 7.270541082164328, 7.284569138276553, 7.298597194388778, 7.312625250501002, 7.326653306613227, 7.340681362725451, 7.354709418837675, 7.3687374749499, 7.382765531062124, 7.396793587174349, 7.410821643286573, 7.424849699398798, 7.438877755511022, 7.452905811623246, 7.466933867735471, 7.480961923847696, 7.49498997995992, 7.509018036072145, 7.5230460921843685, 7.537074148296593, 7.551102204408818, 7.565130260521042, 7.579158316633267, 7.593186372745491, 7.6072144288577155, 7.62124248496994, 7.635270541082164, 7.649298597194389, 7.663326653306613, 7.677354709418838, 7.6913827655310625, 7.705410821643286, 7.719438877755511, 7.733466933867735, 7.74749498997996, 7.761523046092185, 7.775551102204409, 7.789579158316633, 7.803607214428858, 7.817635270541082, 7.831663326653307, 7.845691382765531, 7.859719438877756, 7.87374749498998, 7.887775551102204, 7.901803607214429, 7.915831663326653, 7.929859719438878, 7.943887775551103, 7.9579158316633265, 7.971943887775551, 7.985971943887775, 8], 
    'y': [-1.7678354999999994, -1.6662581950852284, -1.5853590415610577, -1.5243455907450294, -1.4824253939546785, -1.4588060025075436, -1.4526949677211636, -1.4632998409130755, -1.4898281734008183, -1.5314875165019282, -1.5874854215339447, -1.6570294398144063, -1.7393271226608495, -1.833586021390814, -1.9390136873218355, -2.0548176717714526, -2.1802055260572075, -2.314384801496634, -2.4565630494072677, -2.6059478211066507, -2.7617466679123215, -2.9231671411418163, -3.0894167921126736, -3.2597031721424305, -3.433233832548625, -3.6092163246487985, -3.786858199760487, -3.9653859723783493, -4.14419865694757, -4.322787347130546, -4.50064400043701, -4.677260574376714, -4.852129026459394, -5.024741314194801, -5.194589395092677, -5.361165226662766, -5.5239607664148105, -5.682467971858551, -5.836178800503739, -5.984585209860113, -6.127179157437421, -6.2634526007454, -6.3928974972937995, -6.515005804592364, -6.629269480150834, -6.735180481478955, -6.832230766086467, -6.919912291483123, -6.997717015178656, -7.06513689468282, -7.121663887505351, -7.1667899511559945, -7.200007043144499, -7.220824749080558, -7.229212968031671, -7.225634629696312, -7.2105768449800145, -7.1845267247883235, -7.1479713800267755, -7.101397921600916, -7.045293460416275, -6.980145107378398, -6.9064399733928274, -6.824665169365096, -6.735307806200749, -6.638854994805324, -6.535793846084358, -6.4266114709433975, -6.311794980287977, -6.19183148502364, -6.067208096055923, -5.938411924290365, -5.805930080632504, -5.670249675987888, -5.531857821262051, -5.391241627360531, -5.2488882051888766, -5.105284665652611, -4.960918119657292, -4.81627515853643, -4.671745676399823, -4.527510082413869, -4.383721386439255, -4.240532598336692, -4.098096727966878, -3.956566785190505, -3.816095779868288, -3.676836721860918, -3.538942621029095, -3.4025664872335235, -3.2678613303348962, -3.1349801601939227, -3.0040759866712983, -2.8753018196277202, -2.7488106689239027, -2.624755544420527, -2.5032894559783045, -2.3845654134579353, -2.2687364267201136, -2.15595550562555, -2.0463756600349323, -1.9401498998089703, -1.8374312348083621, -1.7383726748938013, -1.6431272299259976, -1.551847909765649, -1.4646598287769028, -1.3815574009735592, -1.302496794074763, -1.2274341710164864, -1.1563256947346963, -1.0891275281653585, -1.0257958342444469, -0.9662867759079206, -0.910556516091755, -0.8585612177319151, -0.8102570437643695, -0.765600157125089, -0.7245467207500367, -0.6870528975751837, -0.6530748505364989, -0.6225687425699471, -0.5954907366115008, -0.5717969955971243, -0.5514436824627882, -0.5343869601444595, -0.5205829915781068, -0.5099879396996976, -0.5025579674451996, -0.4982492377505827, -0.4970179135518136, -0.49882015778486055, -0.5036207415414145, -0.5114783929337514, -0.52250967575308, -0.5368320001090612, -0.5545627761113567, -0.5758194138696258, -0.6007193234935297, -0.6293799150927302, -0.6619185987768854, -0.6984527846556587, -0.7390998828387093, -0.783977303435698, -0.8332024565562863, -0.8868927523101331, -0.9451656008069, -1.008138412156247, -1.0759285964678371, -1.1486535638513304, -1.2264307244163843, -1.3093774882726614, -1.397611265529823, -1.4912494662975289, -1.5904095006854435, -1.6952087788032206, -1.8057647107605255, -1.9221947066670169, -2.0446066571632695, -2.1727805703949095, -2.306093438859129, -2.443897508152028, -2.5855450238697038, -2.7303882316082593, -2.877779376963786, -3.0270707055323838, -3.17761446291015, -3.3287628946931904, -3.4798682464775945, -3.630282763859461, -3.7793586924348928, -3.9264482777999845, -4.0709037655508435, -4.212077401283553, -4.3493214305942205, -4.481988099078942, -4.609429652333818, -4.730998335954946, -4.846046395538423, -4.953926076680343, -5.053989624976813, -5.145589286023931, -5.228077305417787, -5.3008059287544835, -5.36312818248145, -5.414701914268674, -5.455951102621732, -5.487418184084855, -5.509645595202294, -5.5231757725182895, -5.528551152577078, -5.526314171922911, -5.517007267100023, -5.50117287465266, -5.479353431125061, -5.452091373061473, -5.419929137006136, -5.3834091595032945, -5.343073877097183, -5.299465726332055, -5.253127143752144, -5.204600565901695, -5.154428429324951, -5.103153170566153, -5.051317226169542, -4.999463032679364, -4.94813302663986, -4.897869644595271, -4.8492153230898385, -4.802712498667808, -4.758903607873418, -4.718228347775956, -4.680561714288244, -4.64558570869367, -4.612982164981559, -4.582432917141216, -4.553619799161966, -4.526224645033127, -4.499929288744009, -4.474415564283931, -4.449365305642212, -4.424460346808168, -4.399382521771111, -4.373813664520363, -4.347435609045238, -4.319930189335053, -4.290979239379123, -4.260264593166767, -4.227468084687301, -4.192271547930039, -4.154356816884303, -4.113405725539403, -4.069100107884658, -4.021121797909391, -3.969152629602912, -3.912874436954534, -3.8519690539535776, -3.7861520512569564, -3.7155856784670416, -3.6407485445390186, -3.562125955000397, -3.4802032153786735, -3.39546563120136, -3.308398507995965, -3.219487151289992, -3.129216866610934, -3.03807295948632, -2.9465407354436444, -2.8551055000104077, -2.7642525587141225, -2.674467217082298, -2.5862347806424304, -2.5000405549220313, -2.416369845448612, -2.3357079577496673, -2.2585401973527093, -2.185351869785247, -2.116628280574782, -2.0528547352488173, -1.9945165393348625, -1.9420989983604269, -1.8960874178530103, -1.85696710334012, -1.8252152893868643, -1.8009279222937242, -1.783662593991517, -1.7729362357025131, -1.768265778648978, -1.7691681540531845, -1.7751602931374006, -1.7857591271238942, -1.8004815872349393, -1.818844604692802, -1.8403651107197527, -1.8645600365380597, -1.8909463133699929, -1.919040872437821, -1.9483606449638176, -1.9784225621702465, -2.008743555279379, -2.0388405555134854, -2.068230494094837, -2.0964303022457, -2.1229569111883437, -2.1473272521450406, -2.1690582563380563, -2.187666854989663, -2.2026699793221294, -2.2135845605577247, -2.2199277078897524, -2.221421842345716, -2.218390126178684, -2.2112650181030284, -2.2004789768331303, -2.186464461083366, -2.1696539295681077, -2.1504798410017383, -2.129374654098626, -2.1067708275731545, -2.083100820139699, -2.058797090512632, -2.0342920974063357, -2.0100182995351803, -1.9864081556135464, -1.9638941243558106, -1.9429086644763456, -1.9238842346895333, -1.9072532937097457, -1.8934483002513616, -1.8829017130287569, -1.876045990756307, -1.8733135921483897, -1.8751369759193803, -1.881948600783657, -1.8941809254555952, -1.9122664086495709, -1.9365222482506086, -1.9665201893531015, -2.0015353830227913, -2.0408422205494365, -2.0837150932228043, -2.129428392332656, -2.17725650916875, -2.2264738350208564, -2.27635476117873, -2.3261736789321423, -2.375204979570851, -2.422723054384617, -2.4680022946632088, -2.5103170916963835, -2.5489418367739085, -2.583150921185545, -2.612218736221054, -2.6354196731702006, -2.652028123322747, -2.6613184779684547, -2.6625651283970875, -2.655042465898409, -2.638024881762181, -2.610786767278164, -2.5726025137361272, -2.5227465124258264, -2.460552064154095, -2.3863118406457495, -2.3010990315320154, -2.2060094873552614, -2.102139058657842, -1.9905835959821352, -1.8724389498704912, -1.748800970865283, -1.620765509508884, -1.4894284163436424, -1.355885541911941, -1.2212327367561278, -1.086565851418575, -0.9529807364416554, -0.8215732423677213, -0.6934392197391444, -0.5696745190982959, -0.45137499098752887, -0.33963648594921825, -0.23555485452572, -0.14022594725940296, -0.05474561469263994, 0.0197902926322171, 0.08228592417279046, 0.13164542938672597, 0.16677295773165168, 0.18658403983820238, 0.19077197251583877, 0.18029357353928344, 0.1562214219660243, 0.11962809685354969, 0.07158617725934846, 0.01316824224090718, -0.05455312914428667, -0.1305053578387403, -0.21361586478497374, -0.30281207092548845, -0.3970213972028094, -0.4951712645594424, -0.5961890939378907, -0.6990023062806827, -0.8025383225303155, -0.9057245636293132, -1.007488450520184, -1.1067574041454313, -1.2024588454475806, -1.293520195369132, -1.3788688748526083, -1.457432304840516, -1.528137906275363, -1.58991310009967, -1.6416853072559427, -1.6823819929951778, -1.711284913189382, -1.7288865529775925, -1.7359378045651008, -1.733189560157197, -1.721392711959171, -1.7012981521763124, -1.6736567730139136, -1.639219466677262, -1.5987371253716482, -1.5529606413023667, -1.5026409066747008, -1.4485288136939478, -1.391375254565392, -1.331931121494325, -1.2709473066860415, -1.2091747023458255, -1.1473642006789726, -1.0862666938907677, -1.0266330741865026, -0.9692142337714728, -0.914761064850961, -0.8640244596302601, -0.8177553103146654, -0.7767045091094588, -0.741622948219937, -0.7132615198513859, -0.6922856895609504, -0.6787077823935472, -0.6722386369330295, -0.6725874718208886, -0.6794635056986167, -0.6925759572077037, -0.7116340449896406, -0.7363469876859188, -0.7664240039380277, -0.8015743123874597, -0.8415071316757072, -0.8859316804442561, -0.9345571773346029, -0.9870928409882331, -1.0432478900466433, -1.1027315431513232, -1.1652530189437582, -1.230521536065447, -1.2982463131578745, -1.3681365688625315, -1.4399015218209148, -1.5132503906745072, -1.5878923940648098, -1.6635367506333072, -1.7398926790214855, -1.816669397870846, -1.89358071610073, -1.9704340338057897, -2.0471241316100057, -2.123549136449922, -2.1996071752620976, -2.275196374983067, -2.3502148625493904, -2.4245607648976097, -2.4981322089642655, -2.5708273216859183, -2.642544229999105, -2.7131810608403817, -2.782635941146293, -2.85080699785338, -2.9175923578982044, -2.982890148217302, -3.0465984957472196, -3.1086155274245133, -3.168839370185723, -3.2271681509674046, -3.2834999967060994, -3.3377330343383527, -3.389765390800723, -3.4394951930297433, -3.486820567961975, -3.531639642533957, -3.5738510744118046, -3.613410092787637, -3.650377907179328, -3.6848273776802123, -3.7168313643836157, -3.7464627273828754, -3.7737943267713225, -3.798899022642285, -3.8218496750891, -3.8427191442050934, -3.861580290083599, -3.878505972817952, -3.893569052501479, -3.906842389227511, -3.918398843089384, -3.928311274180426, -3.9366525425939747, -3.9434955084233527, -3.9489130317618963, -3.9529779727029384, -3.95576319133981, -3.9573415477658402, -3.957785902074361, -3.957169114358706, -3.955564044712207, -3.9530435532281953, -3.949680499999999], 
    'fillcolor': '#a87c79'
    }

    layout = \
    Layout(
        title         = f'<b>Historical Risk Stream of {root} in {name}<b>',
    )

    return [upFigure(traces = traces, layout = layout, margin = (60,40,40,240))]

def main() :

    dash.run_server(debug = True, host = '0.0.0.0', port = 81)

if  __name__ == '__main__' :

    main()

# region junk
# @dash.callback(
#     [
#         O('manda-wrap', 'hidden'),
#         O('stock-mask', 'hidden'),

#         O('risks-wrap', 'hidden'),
#         O('world-mask', 'hidden'),
#         O('total-mask', 'hidden'),
#         O('burst-mask', 'hidden'),
#         O('trend-mask', 'hidden'),

#         O('views-pick', 'label' ),
#     ],
#     [
#         I('manda-pick', 'n_clicks'),
#         I('risks-pick', 'n_clicks')
#     ]
# )
# def pickView(manda, risks) :

#     manda = manda or 0
#     risks = risks or 0

#     print(f'pickView {manda}:{pickView.manda}, {risks}:{pickView.risks}')

#     if  risks > pickView.risks :
#         pickView.risks    = risks
#         pickView.current  = 'risks'

#     if  manda > pickView.manda :
#         pickView.manda    = manda
#         pickView.current  = 'manda'

#     if  pickView.current == 'risks' : print('pickView Risks View') ; return [T, T] + [F, F, F, F, F] + ['Corporate Global Risks']
#     else                            : print('pickView MandA View') ; return [F, F] + [T, T, T, T, T] + ['Mergers & Acquisitions']

# pickView.current = 'risks'
# pickView.risks   = 0
# pickView.manda   = 0
# endregion