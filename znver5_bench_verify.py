#!/usr/bin/env python3
"""
Verify llvm-exegesis measurements against X86ScheduleZnver5.td scheduling model.
"""

import yaml
import json
import subprocess
import sys
import glob
from pathlib import Path
from typing import Dict, Tuple, List
from statistics import mean, stdev, median

def parse_single_measurement_file(measurements_file: str, verbose: bool = False) -> Dict:
    """Parse a single llvm-exegesis YAML output file"""
    if verbose:
        print(f"Parsing {measurements_file}...")

    if not Path(measurements_file).exists():
        if verbose:
            print(f"ERROR: File {measurements_file} not found!")
        return {}

    measurements = {}
    errors = []
    count = 0

    with open(measurements_file, 'r') as f:
        content = f.read()

        # llvm-exegesis outputs multiple YAML documents separated by ---
        docs = content.split('\n---\n')

        for i, doc_str in enumerate(docs):
            if not doc_str.strip():
                continue

            try:
                doc = yaml.safe_load(doc_str)
                count += 1

                if not doc:
                    continue

                if 'key' in doc:
                    if 'instructions' in doc['key']:
                        # Extract first instruction
                        instr_line = doc['key']['instructions'][0]
                        opcode = instr_line.split()[0]

                        # Look for measurements
                        if 'measurements' in doc:
                            for m in doc['measurements']:
                                if m['key'] == 'inverse_throughput':
                                    measurements[opcode] = {
                                        'inverse_throughput': m['value'],
                                        'per_snippet_value': m.get('per_snippet_value', 0)
                                    }
                                    break
                                elif m['key'] == 'latency':
                                    if opcode not in measurements:
                                        measurements[opcode] = {}
                                    measurements[opcode]['latency'] = m['value']

                # Check for error entries
                if 'error' in doc and doc['error']:
                    error_msg = doc['error']
                    if 'Illegal' in error_msg or 'Cannot' in error_msg:
                        errors.append(error_msg[:100])

            except yaml.YAMLError as e:
                if verbose and count <= 10:
                    print(f"YAML parse error in document {i}: {str(e)[:100]}")
            except Exception as e:
                # Skip malformed entries
                if verbose and count <= 10:
                    print(f"Error in document {i}: {str(e)[:100]}")
                continue

    if verbose:
        print(f"  Found {len(measurements)} valid measurements, {len(errors)} errors")

    return measurements

def filter_outliers_mad(values: List[float], threshold: float = 3.5) -> List[float]:
    """
    Filter outliers using MAD (Median Absolute Deviation) method.
    More robust than standard deviation for small samples with outliers.

    Args:
        values: List of measurement values
        threshold: Number of MADs away from median to consider outlier (default 3.5)

    Returns:
        Filtered list of values with outliers removed
    """
    if len(values) <= 2:
        return values  # Not enough data to filter

    med = median(values)
    # Calculate MAD: median of absolute deviations from median
    abs_deviations = [abs(v - med) for v in values]
    mad = median(abs_deviations)

    if mad == 0:
        # All values are identical or very close
        return values

    # Modified z-score using MAD
    # A value is an outlier if: |value - median| / MAD > threshold
    filtered = []
    outliers = []
    for v in values:
        if abs(v - med) / mad <= threshold:
            filtered.append(v)
        else:
            outliers.append(v)

    # Always keep at least half the values
    if len(filtered) < len(values) // 2:
        return values

    return filtered

def parse_multiple_measurements(pattern: str) -> Dict:
    """Parse multiple measurement files and calculate averages"""
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return {}

    print(f"Found {len(files)} measurement files:")
    for f in files:
        print(f"  - {Path(f).name}")

    # Collect all measurements per instruction
    all_measurements = {}

    for file_path in files:
        print(f"parsing: {file_path}")
        file_measurements = parse_single_measurement_file(file_path)

        for instr, meas in file_measurements.items():
            if instr not in all_measurements:
                all_measurements[instr] = {
                    'inverse_throughput_values': [],
                    'latency_values': []
                }

            if 'inverse_throughput' in meas:
                all_measurements[instr]['inverse_throughput_values'].append(meas['inverse_throughput'])

            if 'latency' in meas:
                all_measurements[instr]['latency_values'].append(meas['latency'])

    # Calculate averages and statistics
    averaged_measurements = {}

    for instr, values in all_measurements.items():
        result = {}

        # Process inverse throughput
        if values['inverse_throughput_values']:
            throughput_vals_raw = values['inverse_throughput_values']
            # Apply outlier filtering
            throughput_vals = filter_outliers_mad(throughput_vals_raw)

            result['inverse_throughput'] = mean(throughput_vals)
            result['inverse_throughput_samples'] = len(throughput_vals)
            result['inverse_throughput_raw_values'] = throughput_vals_raw  # Keep raw values for debugging
            result['inverse_throughput_filtered_values'] = throughput_vals  # Keep filtered values

            # Track if outliers were removed
            if len(throughput_vals) < len(throughput_vals_raw):
                result['inverse_throughput_outliers_removed'] = len(throughput_vals_raw) - len(throughput_vals)

            if len(throughput_vals) > 1:
                result['inverse_throughput_std'] = stdev(throughput_vals)
                # Calculate coefficient of variation (relative std dev)
                if result['inverse_throughput'] > 0:
                    result['inverse_throughput_cv'] = result['inverse_throughput_std'] / result['inverse_throughput']

        # Process latency
        if values['latency_values']:
            latency_vals_raw = values['latency_values']
            # Apply outlier filtering
            latency_vals = filter_outliers_mad(latency_vals_raw)

            result['latency'] = mean(latency_vals)
            result['latency_samples'] = len(latency_vals)

            # Track if outliers were removed
            if len(latency_vals) < len(latency_vals_raw):
                result['latency_outliers_removed'] = len(latency_vals_raw) - len(latency_vals)

            if len(latency_vals) > 1:
                result['latency_std'] = stdev(latency_vals)
                if result['latency'] > 0:
                    result['latency_cv'] = result['latency_std'] / result['latency']

        if result:  # Only add if we have some measurements
            averaged_measurements[instr] = result

    print(f"\nAggregated measurements for {len(averaged_measurements)} instructions")
    print(f"Sample coverage per file: ~{len(averaged_measurements) / len(files):.0f} instructions")

    # Show statistics for a few sample instructions
    sample_instrs = ['IMUL32rr', 'ADD32rr', 'VFMADD213PSZr', 'IMUL8r']
    print("\nSample averaged measurements:")
    for instr in sample_instrs:
        if instr in averaged_measurements:
            meas = averaged_measurements[instr]
            if 'inverse_throughput' in meas:
                cv_str = f", CV={meas.get('inverse_throughput_cv', 0):.2%}" if 'inverse_throughput_cv' in meas else ""
                print(f"  {instr}: inverse_throughput={meas['inverse_throughput']:.3f} (n={meas.get('inverse_throughput_samples', 0)}{cv_str})")

    return averaged_measurements

def load_sched_info() -> Tuple[Dict, Dict]:
    """
    Run llvm-tblgen to generate scheduling info and parse the JSON output.
    Returns: (schedule_info, instr_to_write_class)
    """
    print("\nGenerating scheduling info using llvm-tblgen...")

    # Run llvm-tblgen command
    cmd = [
        './build/bin/llvm-tblgen',
        '-I', './llvm/lib/Target/X86',
        '-I', './llvm/include',
        '-I', './llvm',
        'llvm/lib/Target/X86/X86.td',
        '--gen-sched-info'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        json_data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running llvm-tblgen: {e}")
        print(f"stderr: {e.stderr}")
        return {}, {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output: {e}")
        return {}, {}

    # Extract Znver5Model.SchedWrites
    znver5_model = json_data.get('Znver5Model', {})
    schedule_info = znver5_model.get('SchedWrites', {})
    instruction_mappings = znver5_model.get('InstructionMappings', {})

    # Create instruction to write class mappings from TableGen data
    instr_to_write_class = {}
    for instr_name, instr_info in instruction_mappings.items():
        # Get the first WriteDef if available
        write_defs = instr_info.get('WriteDefs', [])
        if write_defs:
            # Use the first WriteDef as the primary mapping
            instr_to_write_class[instr_name] = write_defs[0]

    print(f"Loaded {len(schedule_info)} scheduling definitions from TableGen")
    print(f"Loaded {len(instr_to_write_class)} instruction-to-write-class mappings from TableGen")

    # Debug: print a few examples
    sample_instrs = ['IMUL32rr', 'ADD32rr', 'VFMADD213PSZr', 'IMUL8r']
    print("Sample instruction mappings:")
    for instr in sample_instrs:
        if instr in instr_to_write_class:
            write_class = instr_to_write_class[instr]
            if write_class in schedule_info:
                sched_info = schedule_info[write_class]
                print(f"  {instr} -> {write_class}: Latency={sched_info.get('Latency')}, InverseThroughput={sched_info.get('InverseThroughput'):.3f}")

    return schedule_info, instr_to_write_class

def compare_schedule_with_measurements(schedule_info: Dict, instr_to_write_class: Dict, measurements: Dict):
    """Compare schedule definitions with actual measurements"""
    print("\n=== Comparing Schedule Info with Measurements ===\n")

    # Track mismatches for summary
    mismatches = []

    # Create reverse mapping: Write class -> list of instructions
    write_class_to_instrs = {}
    for instr, write_class in instr_to_write_class.items():
        if write_class not in write_class_to_instrs:
            write_class_to_instrs[write_class] = []
        write_class_to_instrs[write_class].append(instr)

    # For each schedule definition, find corresponding measurements
    for write_class, sched_def in schedule_info.items():
        # Skip if not a proper schedule definition
        if not isinstance(sched_def, dict) or 'Latency' not in sched_def:
            continue

        # Find instructions that use this Write class
        instrs_using_write = []

        # Direct match in Write classes
        if write_class in write_class_to_instrs:
            instrs_using_write = write_class_to_instrs[write_class]

        # Find measurements for these instructions
        measured_instrs = []
        for instr in instrs_using_write:
            if instr in measurements:
                measured_instrs.append((instr, measurements[instr]))

        if measured_instrs:
            print(f"{write_class}:")

            # Extract expected values from JSON format
            expected_latency = sched_def.get('Latency', 0)
            expected_throughput = sched_def.get('InverseThroughput', 0)

            # Show resource info if available
            resources = sched_def.get('Resources', [])
            if resources:
                res_str = ", ".join([f"{r['Name']}({r.get('NumUnits', 1)}u)" for r in resources[:2]])
                print(f"  Schedule: Latency={expected_latency}, InverseThroughput={expected_throughput:.3f}, Resources=[{res_str}]")
            else:
                print(f"  Schedule: Latency={expected_latency}, InverseThroughput={expected_throughput:.3f}")

            print(f"  Found {len(measured_instrs)} measured instructions:")

            for instr, meas in measured_instrs:
                output = f"    {instr}:"
                has_mismatch = False

                if 'latency' in meas and expected_latency > 0:
                    lat_diff = abs(meas['latency'] - expected_latency)
                    status = "✓" if lat_diff <= 0.5 else "MISMATCH"
                    if status == "MISMATCH":
                        has_mismatch = True
                    output += f" latency={meas['latency']:.2f} (expected={expected_latency}) {status}"

                if 'inverse_throughput' in meas and expected_throughput > 0:
                    thr_diff = abs(meas['inverse_throughput'] - expected_throughput)
                    # Allow 10% tolerance for throughput
                    relative_diff = thr_diff / expected_throughput if expected_throughput > 0 else thr_diff
                    status = "✓" if relative_diff <= 0.1 else "MISMATCH"

                    # Add confidence indicator if we have statistics
                    confidence = ""
                    if 'inverse_throughput_cv' in meas:
                        cv = meas['inverse_throughput_cv']
                        if cv > 0.25:  # High variance (>25% CV)
                            raw_values = meas.get('inverse_throughput_raw_values', [])
                            filtered_values = meas.get('inverse_throughput_filtered_values', raw_values)
                            raw_values_str = ', '.join([f"{v:.3f}" for v in raw_values])
                            filtered_values_str = ', '.join([f"{v:.3f}" for v in filtered_values])
                            if len(filtered_values) < len(raw_values):
                                confidence = f" ⚠️ high variance [raw: {raw_values_str}] [filtered: {filtered_values_str}]"
                            else:
                                confidence = f" ⚠️ high variance [values: {raw_values_str}]"
                        elif 'inverse_throughput_samples' in meas:
                            n = meas['inverse_throughput_samples']
                            outliers_removed = meas.get('inverse_throughput_outliers_removed', 0)
                            if outliers_removed > 0:
                                confidence = f" (n={n}, {outliers_removed} outlier{'s' if outliers_removed > 1 else ''} removed)"
                            else:
                                confidence = f" (n={n})"

                    if status == "MISMATCH":
                        has_mismatch = True
                        mismatches.append({
                            'write_class': write_class,
                            'instruction': instr,
                            'measured': meas['inverse_throughput'],
                            'expected': expected_throughput
                        })
                    output += f" inverse_throughput={meas['inverse_throughput']:.3f} (expected={expected_throughput:.3f}) {status}{confidence}"

                print(output)

            print()

    # Print summary of mismatches
    if mismatches:
        print("\n=== Summary of Throughput Mismatches ===")
        for m in mismatches:
            diff_pct = abs(m['measured'] - m['expected']) / m['expected'] * 100
            print(f"  {m['write_class']}/{m['instruction']}: measured={m['measured']:.3f}, expected={m['expected']:.3f} ({diff_pct:.1f}% diff)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        measurements_pattern = sys.argv[1]
    else:
        # Default to using all znver5_rthroughput files if they exist
        if glob.glob("znver5_rthroughput.*.yaml"):
            measurements_pattern = "znver5_rthroughput.*.yaml"
        else:
            measurements_pattern = "znver5_rthroughput.yaml"

    # Check if pattern contains wildcard or if multiple files match
    if '*' in measurements_pattern or len(glob.glob(measurements_pattern)) > 1:
        print(f"Using multiple measurement files: {measurements_pattern}")
        measurements = parse_multiple_measurements(measurements_pattern)
    else:
        print(f"Using single measurement file: {measurements_pattern}")
        measurements = parse_single_measurement_file(measurements_pattern, verbose=True)

    schedule_info, instr_to_write_class = load_sched_info()

    # Compare schedule definitions with measurements
    if measurements and schedule_info:
        compare_schedule_with_measurements(schedule_info, instr_to_write_class, measurements)