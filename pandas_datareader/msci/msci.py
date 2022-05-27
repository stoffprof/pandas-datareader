import inspect
import json
import numpy as np
import re
import warnings

from pandas import DataFrame, Series, concat, to_numeric, to_datetime

from pandas_datareader._utils import (RemoteDataError, SymbolWarning)
from pandas_datareader.base import _BaseReader
from pandas_datareader.yahoo.headers import DEFAULT_HEADERS
from pandas_datareader.compat import (
    PANDAS_0230,
    string_types,
)


def update_index_codes(**kwargs):
    """
    Update the list of codes for MSCI indexes, which is used as a lookup
    table for required parameters for API request. The data is stored in
    index_codes.json.
    
    """
    return MSCIReader(symbols="", **kwargs)._update_index_codes()


def search(q, **kwargs):
    """
    Regex-based search of index names. Example usage:
        
        >>> from pandas_datareader.msci import msci
        >>> msci.search('netherlands')


    Returns
    -------
    A list of valid index names.
    """
    return MSCIReader(symbols="", **kwargs)._search(q)



class MSCIReader(_BaseReader):
    """
    Returns DataFrame of with historical over date range,
    start to end.
    To avoid being penalized by server, pauses between downloading 'chunks' of 
    symbols can be specified.

    Parameters
    ----------
    symbols : string, array-like object (list, tuple, Series), or DataFrame
        Single index name, or array-like object of index names. Valid index
        names may be found using the search method.
    start : string, int, date, datetime, Timestamp
        Starting date. Parses many different kind of date
        representations (e.g., 'JAN-01-2010', '1/1/10', 'Jan, 1, 1980').
        Defaults to 5 years before current date.
    end : string, int, date, datetime, Timestamp
        Ending date
    retry_count : int, default 3
        Number of times to retry query request.
    pause : int, default 0.1
        Time, in seconds, to pause between consecutive queries. If single value
        given for symbol, represents the pause between retries.
    session : Session, default None
        requests.sessions.Session instance to be used. Passing a session
        is an advanced usage and you must set any required
        headers in the session directly.
    interval : string, default 'd'
        Time interval code, valid values are 'd' for daily, 'm' for monthly,
        'a' for annualy.
        
    Returns
    -------
    pandas.DataFrame
        A DataFrame with requested data.
        
    Examples
    --------
    
    >>> import pandas_datareader as pdr
    >>> pdr.get_data_msci('USA')  # a single index
    >>> pdr.get_data_msci(['USA', 'CANADA'])  # multiple indexes
        
    """

    def __init__(
        self,
        symbols=None,
        start=None,
        end=None,
        retry_count=3,
        pause=0.1,
        session=None,
        interval="d",
    ):
        super().__init__(
            symbols=symbols,
            start=start,
            end=end,
            retry_count=retry_count,
            pause=pause,
            session=session,
        )

        self.pause_multiplier = 2.5
        if session is None:
            self.headers = DEFAULT_HEADERS
        else:
            self.headers = session.headers

        self.interval = interval[0]
        if self.interval not in ["d", "m", "a"]:
            raise ValueError(
                "Invalid interval: valid values are  '(d)aily', '(m)monthly'"
                "and '(a)nnually'."
            )
        self.idxloc = ('/'.join(inspect.getfile(self.__class__).split('/')[:-1])
                       + '/index_codes.json')
        with open(self.idxloc, 'r') as f:
            self.index_codes = json.load(f)
            
    
    @property
    def url(self):
        return "https://app2.msci.com/products"

    @property
    def params(self):
        intervals = {"d": "DAILY", "m": "END_OF_MONTH", "a": "ANNUAL"}

        params = {
            "currency_symbol": "USD",
            "index_variant": "STRD",
            "start_date": self.start.strftime("%Y%m%d"),
            "end_date": self.end.strftime("%Y%m%d"),
            "data_frequency": intervals[self.interval]
        }
        return params

    def _get_params(self, symbol):
        params = self.params
        try:
            params["index_codes"] = self.index_codes[symbol]
            params["symbol"] = symbol
            return params
        except KeyError:
            msg = ("Invalid index name {} using {}. Use `search` method to " +
                    "find allowed values.")
            raise RemoteDataError(msg.format(symbol, self.__class__.__name__))

    def read(self):
        """Read data"""
        if isinstance(self.symbols, (string_types, int)):
            df = self._read_one_data(params=self._get_params(self.symbols))
        else:
            df = self._dl_mult_symbols()
        return df


    def _read_one_data(self, params):
        """Read one data from specified symbol"""
        symbol = params['symbol']
        del params['symbol']
        
        url = self.url + "/service/index/indexmaster/getLevelDataForGraph"
        resp = self._get_response(url, params=params, headers=self.headers)
        
        if resp.ok:
            prices = DataFrame(resp.json()["indexes"]["INDEX_LEVELS"])
            prices.index = to_datetime(prices["calc_date"].astype(str))
            del prices["calc_date"]
            prices.index.name = ""
            prices.columns = [symbol]
            return prices
        else:
            msg = "No data fetched for symbol {} using {}"
            raise RemoteDataError(msg.format(symbol, self.__class__.__name__))


    def _dl_mult_symbols(self):
        stocks = []
        failed = []
        passed = []

        for sym in self.symbols:
            try:
                stocks.append(self._read_one_data(self._get_params(sym)))
                passed.append(sym)
            except (IOError, KeyError):
                msg = "Failed to get data for symbol: {0!r}."
                warnings.warn(msg.format(sym), SymbolWarning)
                failed.append(sym)

        if len(passed) == 0:
            msg = "No data fetched using {0!r}"
            raise RemoteDataError(msg.format(self.__class__.__name__))
        
        if PANDAS_0230:
            result = concat(stocks, sort=True, axis=1)
        else:
            result = concat(stocks, axis=1)
        
        # for sym in failed:
        #     result[sym] = np.nan
            
        return result


    def _update_index_codes(self):
        """
        Get index codes for all available markets and save as JSON file in
        module directory.
        """
        response = self._get_response(self.url + 
               "/index-data-search/resources/index-data-search"
               "/js/data/index-panel-data.json")
        markets = response.json()["market"]
        markets = DataFrame(markets).set_index("id").squeeze()

        # Get list of indexes in each scope & market
        index_params = {
            "index_size": 12,
            "index_style": "None",
            "index_suite": "C"
        }
        idxs = []
        for index_scope in ["Region", "Country"]:
            for index_market in markets.index:
                index_params["index_scope"] = index_scope
                index_params["index_market"] = index_market

                response = self.session.get(self.url + 
                        "/service/index/indexmaster/searchIndexes",
                         params=index_params)

                idx = DataFrame(response.json()["indexes"])
                idx.insert(0, "index_scope", index_scope)
                idx.insert(0, "index_market", markets[index_market])
                idxs.append(idx)

        idxs = concat(idxs).dropna()
    
        idxs["msci_index_code"] = to_numeric(idxs["msci_index_code"],
                                                downcast="integer")
        idxs['index_name'] = idxs['index_name'].str.strip()
        idxs = (idxs[['index_name', 'msci_index_code']]
                    .drop_duplicates()
                    .set_index('index_name')
                    .squeeze()
                    .sort_index()
                    .to_dict()
                )
        with open(self.idxloc, 'w') as f:
            json.dump(idxs, f)
        self.index_codes = idxs
    
    def _search(self, q):
        """ Find index code by name. """
        idxs = Series(self.index_codes)
        mask = idxs.index.str.contains(q, flags=re.IGNORECASE)
        rslt = idxs[mask]
        if len(rslt) > 0:
            return rslt.index.to_list()
        else:
            print("No match found.")
