#!/usr/bin/env python

"""F-Seq Version2 call peaks from signal file main script.

This module provides functionality to call peaks directly from pre-computed
HDF5 signal files (.h5 files) produced by the callpeak command with
-sig_format np_array option.
"""

import time

import h5py
import numpy as np
from pandas import concat, DataFrame

from fseq2 import fseq2


def main(args):
    """Main entry point for callpeak_sig subcommand.

    Args:
        args: argparse.Namespace with signal file path and parameters
    """
    ### 1. Setup and validation ###
    if args.v:
        print('=====================================', flush=True)
        print(f'F-Seq Version {fseq2.__version__}', flush=True)
        print('=====================================', flush=True)
        print('#1: Loading signal file and calculating parameters', flush=True)

    signal_file_path = args.signal_file

    # Open and inspect HDF5 file
    with h5py.File(signal_file_path, 'r') as sig_file:
        chrom_ls = list(sig_file.keys())

        if not chrom_ls:
            raise ValueError(f"No chromosome data found in {signal_file_path}")

        if args.use_file_params:
            # Read parameters from file (as computed by callpeak)
            if 'threshold' not in sig_file.attrs:
                raise ValueError(
                    "Signal file does not contain callpeak parameters. "
                    "Regenerate with a recent version of fseq2 callpeak -sig_format np_array."
                )
            # Core peak calling parameters
            threshold = float(sig_file.attrs['threshold'])
            peak_region_threshold = float(sig_file.attrs['peak_region_threshold'])
            min_distance = int(sig_file.attrs['min_distance'])
            min_prominence = float(sig_file.attrs['min_prominence'])
            window_size = int(sig_file.attrs['window_size'])
            lambda_bg_lower_bound = float(sig_file.attrs['lambda_bg_lower_bound'])
            sparse_data = bool(sig_file.attrs['sparse_data'])
        else:
            # Calculate thresholds from signal (original behavior)
            threshold, peak_region_threshold = calculate_thresholds(
                sig_file, chrom_ls, args
            )
            min_distance = args.min_distance
            min_prominence = None  # Not available without file params
            window_size = args.window_size
            lambda_bg_lower_bound = 0  # Default when calling from signal without params
            sparse_data = False

    if args.v:
        print(f'\tSignal file: {signal_file_path}', flush=True)
        print(f'\tChromosomes found: {len(chrom_ls)}', flush=True)
        print(f'\tThreshold: {threshold:.3f}', flush=True)
        print(f'\tPeak region threshold: {peak_region_threshold:.3f}', flush=True)
        print(f'\tMin distance: {min_distance}', flush=True)
        print(f'\tWindow size: {window_size}', flush=True)
        if args.use_file_params:
            print(f'\tMin prominence: {min_prominence:.3f}', flush=True)
            print(f'\tLambda bg lower bound: {lambda_bg_lower_bound:.3f}', flush=True)
            print(f'\tSparse data: {sparse_data}', flush=True)
        print('#1: Done', flush=True)
        print('-------------------------------------', flush=True)

    ### 2. Process each chromosome ###
    if args.v:
        print('#2: Calling peaks from signal\n', flush=True)

    results = []
    with h5py.File(signal_file_path, 'r') as sig_file:
        for chrom in chrom_ls:
            signal_array = sig_file[chrom][:].astype(np.float32)
            first_cut = int(sig_file.attrs[chrom])

            result_df = process_chrom_from_signal(
                chrom=chrom,
                signal_array=signal_array,
                first_cut=first_cut,
                threshold=threshold,
                peak_region_threshold=peak_region_threshold,
                min_distance=min_distance,
                min_prominence=min_prominence,
                window_size=window_size,
                lambda_bg_lower_bound=lambda_bg_lower_bound,
                sparse_data=sparse_data,
                verbose=args.v
            )

            if result_df is not None and not result_df.empty:
                results.append(result_df)

    if not results:
        print("Warning: No peaks found in any chromosome.", flush=True)
        # Write empty output files
        DataFrame().to_csv(f'{args.o}/{args.name}_summits.narrowPeak',
                          sep='\t', header=None, index=None)
        DataFrame().to_csv(f'{args.o}/{args.name}_peaks.narrowPeak',
                          sep='\t', header=None, index=None)
        return

    result_df = concat(results)

    if args.v:
        print(f'\n\tTotal peaks before filtering: {result_df.shape[0]}', flush=True)

    ### 3. Calculate p-value and q-value ###
    if not args.skip_stats:
        if args.v:
            print('#2: Computing p-value and q-value', flush=True)

        result_df = fseq2.interpolate_poisson_p_value(result_df)
        result_df = fseq2.calculate_q_value(
            result_df=result_df,
            p_thr=args.p_thr,
            q_thr=args.q_thr,
            num_peaks=args.num_peaks
        )
    else:
        # Provide dummy statistical values when skipping stats
        result_df['-log10_p_value_interpolated'] = result_df['score']
        result_df['q_value'] = result_df['score']

    if args.v:
        print(f'\tPeaks after filtering: {result_df.shape[0]}', flush=True)
        print('#2: Done', flush=True)
        print('-------------------------------------', flush=True)

    ### 4. Write output ###
    if args.v:
        print('#3: Writing output', flush=True)

    fseq2.narrowPeak_writer(
        result_df=result_df,
        peak_type='summit',
        name=args.name,
        out_dir=args.o,
        prior_pad_summit=args.prior_pad_summit,
        sort_by=args.sort_by
    )
    fseq2.narrowPeak_writer(
        result_df=result_df,
        peak_type='peak',
        name=args.name,
        out_dir=args.o,
        sort_by=args.sort_by,
        standard_narrowpeak=args.standard_narrowpeak
    )

    if args.v:
        print(f'\tOutput: {args.o}/{args.name}_summits.narrowPeak', flush=True)
        print(f'\tOutput: {args.o}/{args.name}_peaks.narrowPeak', flush=True)
        print(f'#3: Done - {result_df.shape[0]} peaks written', flush=True)
        print('-------------------------------------', flush=True)
        print(f'Thanks for using F-seq{fseq2.__version__}!\n', flush=True)

    return


def calculate_thresholds(sig_file, chrom_ls, args):
    """Calculate or parse threshold values.

    Args:
        sig_file: Open h5py File object
        chrom_ls: List of chromosome names
        args: argparse.Namespace with threshold parameters

    Returns:
        tuple: (threshold, peak_region_threshold)
    """
    # Determine threshold
    if args.t == 'auto':
        # Sample signal to estimate threshold
        all_signals = []
        for chrom in chrom_ls[:min(5, len(chrom_ls))]:  # Sample up to first 5 chromosomes
            sig = sig_file[chrom][:]
            # Subsample for efficiency (every 100th position)
            all_signals.append(sig[::100])

        combined_signal = np.concatenate(all_signals)
        # Use only non-zero positive values for estimation
        combined_signal = combined_signal[combined_signal > 0]

        if combined_signal.size == 0:
            raise ValueError("Signal file contains no positive values - cannot estimate threshold")

        mean_sig = np.mean(combined_signal)
        std_sig = np.std(combined_signal)
        threshold = mean_sig + args.t_std * std_sig
    else:
        threshold = float(args.t)

    # Determine peak region threshold
    if args.tp is None:
        peak_region_threshold = threshold
    elif args.tp == 'auto':
        # Reuse or recalculate from signal
        if 'combined_signal' not in dir():
            all_signals = []
            for chrom in chrom_ls[:min(5, len(chrom_ls))]:
                sig = sig_file[chrom][:]
                all_signals.append(sig[::100])
            combined_signal = np.concatenate(all_signals)
            combined_signal = combined_signal[combined_signal > 0]

        mean_sig = np.mean(combined_signal)
        std_sig = np.std(combined_signal)
        peak_region_threshold = mean_sig + args.tp_std * std_sig
    else:
        peak_region_threshold = float(args.tp)

    return threshold, peak_region_threshold


def process_chrom_from_signal(chrom, signal_array, first_cut, threshold,
                               peak_region_threshold, min_distance, min_prominence,
                               window_size, lambda_bg_lower_bound, sparse_data,
                               verbose=False):
    """Process a single chromosome from signal array.

    Args:
        chrom: Chromosome name
        signal_array: numpy array of signal values (float32)
        first_cut: Starting genomic position
        threshold: Minimum peak height threshold
        peak_region_threshold: Threshold for contiguous peak regions
        min_distance: Minimum distance between peaks
        min_prominence: Minimum prominence for peaks (from original callpeak)
        window_size: Window size for lambda calculation
        lambda_bg_lower_bound: Lower bound for background lambda
        sparse_data: Whether to use sparse data mode for local lambda calculation
        verbose: Whether to print progress

    Returns:
        pd.DataFrame with peak data or empty DataFrame if no peaks found
    """
    start_time = time.time()

    if signal_array.size == 0:
        if verbose:
            print(f'\t{chrom}: No signal data - skipping', flush=True)
        return DataFrame()

    last_cut = first_cut + signal_array.size

    # Call peaks using existing function with all parameters from original callpeak
    result_df = fseq2.call_peaks(
        chrom=chrom,
        first_cut=first_cut,
        kdepy_result=signal_array,
        min_height=threshold,
        peak_region_threshold=peak_region_threshold,
        min_distance=min_distance,
        min_prominence=min_prominence
    )

    if result_df.empty:
        if verbose:
            print(f'\t{chrom}: No peaks found', flush=True)
        return result_df

    # Calculate query_value and lambda_local for statistical testing
    summit_abs_pos_array = result_df['summit'].values - first_cut

    # Calculate query value (average signal around summit)
    query_value = fseq2.calculate_query_value(
        result_df=result_df,
        kdepy_result=signal_array,
        summit_abs_pos_array=summit_abs_pos_array,
        window_size=window_size,
        use_max=False
    )

    # Calculate lambda_bg (background estimate) using the same lower bound as original
    lambda_bg = fseq2.calculate_lambda_bg(
        kdepy_result_control=signal_array,
        window_size=window_size,
        lambda_bg_lower_bound=lambda_bg_lower_bound
    )

    # Calculate lambda_local (local background for each peak) with same sparse_data setting
    lambda_local = fseq2.find_local_lambda(
        control_np_tmp=signal_array,
        control_np_tmp_name=False,  # Signal is in memory, not file
        summit_abs_pos_array=summit_abs_pos_array,
        lambda_bg=lambda_bg,
        window_size=window_size,
        sparse_data=sparse_data,
        use_max=False
    )

    result_df['query_value'] = query_value
    result_df['lambda_local'] = lambda_local

    end_time = time.time()
    if verbose:
        print(f'\t{chrom}: first={first_cut}, last={last_cut}, '
              f'peaks={result_df.shape[0]}, completed in {end_time - start_time:.3f} seconds.',
              flush=True)

    return result_df
