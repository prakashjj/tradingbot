def get_mtf_signal_v2(candles, timeframes, percent_to_min=5, percent_to_max=5):
    signals = {}
    
    # Get the OHLCV data for the 1-minute timeframe
    data_1m = np.array([[c['open'], c['high'], c['low'], c['close'], c['volume']] for c in candles['1m']], dtype=np.double)
    
    # Get the HT sine wave indicator for the 1-minute timeframe
    sine, leadsine = talib.HT_SINE(data_1m[:, 3])
    
    # Normalize the HT sine wave indicator to the minimum and maximum prices in the market data
    min_price = np.nanmin(data_1m[:, 3])
    max_price = np.nanmax(data_1m[:, 3])
    norm_sine = (sine - min_price) / (max_price - min_price)
    
    # Get the minimum and maximum values of the normalized HT Sine Wave indicator
    min_sine = np.nanmin(norm_sine)
    max_sine = np.nanmax(norm_sine)
    
    # Calculate the percentage distance from the current close on sine to the minimum and maximum values of the normalized HT Sine Wave indicator
    close = data_1m[-1][-2]
    percent_to_min_val = (max_sine - norm_sine[-1]) / (max_sine - min_sine) * 100
    percent_to_max_val = (norm_sine[-1] - min_sine) / (max_sine - min_sine) * 100
    
    for timeframe in timeframes:
        # Get the OHLCV data for the given timeframe
        ohlc_data = np.array([[c['open'], c['high'], c['low'], c['close'], c['volume']] for c in candles[timeframe]], dtype=np.double)
        
        # Calculate the momentum signal for the given timeframe
        close_prices = ohlc_data[:, 3]
        momentum = talib.MOM(close_prices, timeperiod=14)
        
        # Calculate the minimum and maximum values for the momentum signal
        min_momentum = np.nanmin(momentum)
        max_momentum = np.nanmax(momentum)
        
        # Calculate the percentage distance from the current momentum to the minimum and maximum values of the momentum signal
        current_momentum = momentum[-1]
        percent_to_min_momentum = (max_momentum - current_momentum) / (max_momentum - min_momentum) * 100
        percent_to_max_momentum = (current_momentum - min_momentum) / (max_momentum - min_momentum) * 100
        
        # Calculate the new momentum signal based on percentages from the MTF signal and the initial momentum signal
        percent_to_min_combined = (percent_to_min_val + percent_to_min_momentum) / 2
        percent_to_max_combined = (percent_to_max_val + percent_to_max_momentum) / 2
        momentum_signal = percent_to_max_combined - percent_to_min_combined
        
        # Calculate the new average for the MTF signal based on the percentage distance from the current close to the minimum and maximu values of the normalized HT Sine Wave indicator, and the given percentage thresholds
        min_mtf = np.nanmin(ohlc_data[:, 3])
        max_mtf = np.nanmax(ohlc_data[:, 3])
        percent_to_min_custom = percent_to_min / 100
        percent_to_max_custom = percent_to_max / 100
        min_threshold = min_mtf + (max_mtf - min_mtf) * percent_to_min_custom
        max_threshold = max_mtf - (max_mtf - min_mtf) * percent_to_max_custom
        filtered_close = np.where(ohlc_data[:, 3] < min_threshold, min_threshold, ohlc_data[:, 3])
        filtered_close = np.where(filtered_close > max_threshold, max_threshold, filtered_close)
        avg_mtf = np.nanmean(filtered_close)
        
        # Store the signals for the given timeframe
        signals[timeframe] = {'momentum': momentum_signal, 'ht_sine_percent_to_min': percent_to_min_val, 'ht_sine_percent_to_max': percent_to_max_val, 'mtf_average': avg_mtf, 'min_threshold': min_threshold, 'max_threshold': max_threshold}
    
    current_time = datetime.datetime.utcnow() + datetime.timedelta(hours=3)

    # Print the results
    print("Current time:", current_time.strftime('%Y-%m-%d %H:%M:%S'))
    print(f"HT Sine Wave Percent to Min: {percent_to_min_val:.2f}%")
    print(f"HT Sine Wave Percent to Max: {percent_to_max_val:.2f}%")
    print(f"Momentum Percent to Min: {percent_to_min_momentum:.2f}%")
    print(f"Momentum Percent to Max: {percent_to_max_momentum:.2f}%")
    print(f"Combined Percent to Min: {percent_to_min_combined:.2f}%")
    print(f"Combined Percent to Max: {percent_to_max_combined:.2f}%")
    print(f"New Momentum Signal: {momentum_signal:.2f}")
    print(f"New MTF Average:")
    for timeframe in timeframes:
        print(f"{timeframe}: {signals[timeframe]['mtf_average']:.2f} (min threshold: {signals[timeframe]['min_threshold']:.2f}, max threshold: {signals[timeframe]['max_threshold']:.2f})")
    print()

    return signals

get_mtf_signal_v2(candles, timeframes, percent_to_min=5, percent_to_max=5)

def entry_long(quantity, symbol):
    order = client.futures_create_order(
        symbol=symbol,
        side='BUY',
        type='MARKET',
        quantity=quantity
    )
    print(f'Long entry order placed: {order}')

def entry_short(quantity, symbol):
    order = client.futures_create_order(
        symbol=symbol,
        side='SELL',
        type='MARKET',
        quantity=quantity
    )
    print(f'Short entry order placed: {order}')

def exit_trade():
    order = client.futures_create_order(
        symbol=symbol,
        side='SELL' if side == 'long' else 'BUY',
        type='MARKET',
        quantity=abs(float(client.futures_position_information(symbol=symbol)[0]['positionAmt']))
    )
    print(f'Exit order placed: {order}')

def main():
    # Variables
    global closed_positions
    global TRADE_SYMBOL
    global TRADE_TYPE
    global TRADE_LVRG
    global STOP_LOSS_THRESHOLD
    global TAKE_PROFIT_THRESHOLD
    global BUY_THRESHOLD
    global SELL_THRESHOLD
    global EMA_SLOW_PERIOD
    global EMA_FAST_PERIOD

    # Initialize variables for tracking trade state
    trade_open = False
    trade_side = None
    trade_entry_pnl = 0
    trade_exit_pnl = 0
    trade_entry_time = 0

    while True:
        try:
            # Check if balance is zero
            account_balance = float(get_account_balance())
            if account_balance == 0:
                print("Balance is zero. Exiting program.")
                break

            # Define start and end time for historical data
            start_time = int(time.time()) - (1800 * 4)  # 60-minute interval (4 candles)
            end_time = int(time.time())

            # Define the candles and timeframes to use for the signals
            candles = get_historical_candles(TRADE_SYMBOL, start_time, end_time, '1m')
            timeframes = ['1m', '3m', '5m']

            # Get the MTF signal
            signals = get_mtf_signal_v2(candles, timeframes, percent_to_min=1, percent_to_max=1)

            # Check if the '1m' key exists in the signals dictionary
            if '1m' in signals:
                # Check if the combined percent to min/max signal keys exist
                if 'combined_percent_to_min' in signals['1m'] and 'combined_percent_to_max' in signals['1m']:
                    percent_to_min_combined = signals['1m']['combined_percent_to_min']
                    percent_to_max_combined = signals['1m']['combined_percent_to_max']
                    percent_to_min_val = signals['1m']['ht_sine_percent_to_min']
                    percent_to_max_val = signals['1m']['ht_sine_percent_to_max']
                    percent_to_min_momentum = signals['1m']['momentum_percent_to_min']
                    percent_to_max_momentum = signals['1m']['momentum_percent_to_max']
                    mtf_average = signals['1m']['mtf_average']

                    # Calculate the slow and fast EMA
                    ema_slow = calculate_ema(candles, EMA_SLOW_PERIOD)
                    ema_fast = calculate_ema(candles, EMA_FAST_PERIOD)

                    # Check if the price closes below the fast EMA and the fast EMA is below the slow EMA and the HT Sine Wave Percent to Min is less than 10 and less than the HT Sine Wave Percent to Max and the MTF average is above the close price for a long trade
                    if candles[-1]['close'] < ema_fast[-1] and ema_fast[-1] < ema_slow[-1] and percent_to_min_val < 25 and percent_to_min_val < percent_to_max_val and mtf_average > candles[-1]['close']:
                        # Place a long trade
                        if not trade_open:
                            entry_long(TRADE_SYMBOL)
                            trade_open = True
                            trade_side = 'long'
                            trade_entry_pnl = float(client.futures_position_information(symbol=TRADE_SYMBOL)[0]['unRealizedProfit'])
                            trade_exit_pnl = 0
                            trade_entry_time = int(time.time())
                            print(f"Entered long trade at {trade_entry_time}")
                        else:
                            print("Trade already open.")

                    # Check if the price closes above the fast EMA and the fast EMA is above the slow EMA and the HT Sine Wave Percent to Min is greater than 90 and greater than the HT Sine Wave Wave Percent to Max and the MTF average is below the close price for a short trade
                    elif candles[-1]['close'] > ema_fast[-1] and ema_fast[-1] > ema_slow[-1] and percent_to_max_val < 25 and percent_to_min_val > percent_to_max_val and mtf_average < candles[-1]['close']:
                        # Place a short trade
                        if not trade_open:
                            entry_short(TRADE_SYMBOL)
                            trade_open = True
                            trade_side = 'short'
                            trade_entry_pnl = float(client.futures_position_information(symbol=TRADE_SYMBOL)[0]['unRealizedProfit'])
                            trade_exit_pnl = 0
                            trade_entry_time = int(time.time())
                            print(f"Entered short trade at {trade_entry_time}")
                        else:
                            print("Trade already open.")

                    # Check if the trade has exceeded the stop loss threshold
                    if trade_open and abs(float(client.futures_position_information(symbol=TRADE_SYMBOL)[0]['unRealizedProfit'])) >= STOP_LOSS_THRESHOLD:
                        # Exit the trade
                        exit_trade()
                        trade_open = False
                        trade_exit_pnl = float(client.futures_position_information(symbol=TRADE_SYMBOL)[0]['unRealizedProfit'])
                        print(f"Exited trade at stop loss threshold {int(time.time())}")

                        # Enter a new trade with reversed side
                        if trade_side == 'long':
                            entry_short(TRADE_SYMBOL)
                            trade_side = 'short'
                        elif trade_side == 'short':
                            entry_long(TRADE_SYMBOL)
                            trade_side = 'long'

                        # Reset trade variables
                        trade_open = True
                        trade_entry_pnl = float(client.futures_position_information(symbol=TRADE_SYMBOL)[0]['unRealizedProfit'])
                        trade_exit_pnl = 0
                        trade_entry_time = int(time.time())

                    # Check if the trade has exceeded the take profit threshold
                    elif trade_open and abs(float(client.futures_position_information(symbol=TRADE_SYMBOL)[0]['unRealizedProfit'])) >= TAKE_PROFIT_THRESHOLD:
                        # Exit the trade
                        exit_trade()
                        trade_open = False
                        trade_exit_pnl = float(client.futures_position_information(symbol=TRADE_SYMBOL)[0]['unRealizedProfit'])
                        print(f"Exited trade at take profit threshold {int(time.time())}")

                        # Enter a new trade with reversed side
                        if trade_side == 'long':
                            entry_short(TRADE_SYMBOL)
                            trade_side = 'short'
                        elif trade_side == 'short':
                            entry_long(TRADE_SYMBOL)
                            trade_side = 'long'

                        # Reset trade variables
                        trade_open = True
                        trade_entry_pnl = float(client.futures_position_information(symbol=TRADE_SYMBOL)[0]['unRealizedProfit'])
                        trade_exit_pnl = 0
                        trade_entry_time = int(time.time())

                    # Print the signal values for debugging purposes
                    print(f"HT Sine Wave Percent to Min: {percent_to_min_val}, HT Sine Wave Percent to Max: {percent_to_max_val}, Momentum Percent to Min: {percent_to_min_momentum}, Momentum Percent to Max: {percent_to_max_momentum}")
                    print(f"Combined Percent to Min: {percent_to_min_combined}, Combined Percent to Max: {percent_to_max_combined}")
                    print(f"MTF Average: {mtf_average}")
                    print(f"Fast EMA: {ema_fast[-1]}, Slow EMA: {ema_slow[-1]}")
                    print(f"Unrealized PNL: {float(client.futures_position_information(symbol=TRADE_SYMBOL)[0]['unRealizedProfit'])}")

            # Wait for 5sec before checking again
            time.sleep(5)

        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            time.sleep(5)
            continue
