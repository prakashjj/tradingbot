def main():
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
            timeframes = ['1m']

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
                    new_momentum_signal = signals['1m']['momentum_signal']

                    # Check if the HT Sine Wave Percent to Min is less than 10 and less than the HT Sine Wave Percent to Max and the MTF average is above the close price and the close price is closer to the percent to min threshold for a long trade and the new momentum signal is positive
                    if percent_to_min_val < 10 and percent_to_min_val < percent_to_max_val and mtf_average > candles[-1]['close'] and candles[-1]['close'] - percent_to_min_val < percent_to_max_val - candles[-1]['close'] and momentum_signal > 0:
                        print("BUY signal")

                    # Check if the HT Sine Wave Percent to Min is greater than 90 and greater than the HT Sine Wave Wave Percent to Max and the MTF average is below the close price and the close price is closer to the percent to max threshold for a short trade and the new momentum signal is negative
                    elif percent_to_max_val < 10 and percent_to_min_val > percent_to_max_val and mtf_average < candles[-1]['close'] and percent_to_max_val - candles[-1]['close'] < candles[-1]['close'] - percent_to_min_val and momentum_signal < 0:
                        print("SELL signal")

                    # Print the signal values for debugging purposes
                    print(f"HT Sine Wave Percent to Min: {percent_to_min_val}, HT Sine Wave Percent to Max: {percent_to_max_val}, Momentum Percent to Min: {percent_to_min_momentum}, Momentum Percent to Max: {percent_to_max_momentum}")
                    print(f"Combined Percent to Min: {percent_to_min_combined}, Combined Percent to Max: {percent_to_max_combined}, MTF Average: {mtf_average}")
                    print(f"New Momentum Signal: {new_momentum_signal}")
                    print(f"New MTF Average:\n1m: {signals['1m']['new_mtf_average']} (min threshold: {signals['1m']['min_threshold']}, max threshold: {signals['1m']['max_threshold']})")
                    print("")

            # Wait for the next iteration
            time.sleep(5)

        except Exception as e:
            print(f"An error occurred: {e}")
            continue
