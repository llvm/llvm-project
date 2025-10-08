#!/usr/bin/env python3
"""
Verify llvm-exegesis measurements against X86ScheduleZnver5.td scheduling model.
"""

import yaml
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

def parse_measurements_test(measurements_file: str):
    """Test parsing the llvm-exegesis YAML output"""
    print(f"Testing parse of {measurements_file}...")

    if not Path(measurements_file).exists():
        print(f"ERROR: File {measurements_file} not found!")
        return

    measurements = {}
    errors = []
    count = 0

    with open(measurements_file, 'r') as f:
        content = f.read()

        # llvm-exegesis outputs multiple YAML documents separated by ---
        docs = content.split('\n---\n')

        print(f"Found {len(docs)} YAML documents in file")

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
                print(f"YAML parse error in document {i}: {str(e)[:100]}")
            except Exception as e:
                # Skip malformed entries
                if count <= 10:
                    print(f"Error in document {i}: {str(e)[:100]}")
                continue

    print(f"\n=== Parsing Summary ===")
    print(f"Successfully parsed: {len(measurements)} instructions")
    print(f"Errors/failures: {len(errors)} instructions")

    return measurements

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

    # Create instruction to write class mappings
    # This is a simplified mapping - in production you would parse InstRW from TableGen
    instr_to_write_class = {
        # Integer multiply
        'IMUL8r': 'WriteIMul8',
        'IMUL16rr': 'WriteIMul16Reg',
        'IMUL16rm': 'WriteIMul16RegLd',
        'IMUL32rr': 'WriteIMul32Reg',
        'IMUL32rm': 'WriteIMul32RegLd',
        'IMUL64rr': 'WriteIMul64Reg',
        'IMUL64rm': 'WriteIMul64RegLd',

        # Basic ALU
        'ADD32rr': 'WriteALU',
        'ADD32rm': 'WriteALULd',
        'ADD64rr': 'WriteALU',
        'SUB32rr': 'WriteALU',
        'XOR32rr': 'WriteALU',
        'CMP32rr': 'WriteALU',

        # FMA instructions
        'VFMADD213PSr': 'WriteFMA',
        'VFMADD213PSYr': 'WriteFMAY',
        'VFMADD213PSZr': 'WriteFMAZ',
        'VFMADD213PDr': 'WriteFMA',
        'VFMADD213PDYr': 'WriteFMAY',
        'VFMADD213PDZr': 'WriteFMAZ',
    }

    print(f"Loaded {len(schedule_info)} scheduling definitions from TableGen")
    print(f"Using {len(instr_to_write_class)} instruction-to-write-class mappings")

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
                    if status == "MISMATCH":
                        has_mismatch = True
                        mismatches.append({
                            'write_class': write_class,
                            'instruction': instr,
                            'measured': meas['inverse_throughput'],
                            'expected': expected_throughput
                        })
                    output += f" throughput={meas['inverse_throughput']:.3f} (expected={expected_throughput:.3f}) {status}"

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
        measurements_file = sys.argv[1]
    else:
        measurements_file = "znver5_rthroughput.yaml"
        # measurements_file = "znver5_latency.yaml"

    measurements = parse_measurements_test(measurements_file)
    schedule_info, instr_to_write_class = load_sched_info()

    # Compare schedule definitions with measurements
    if measurements and schedule_info:
        compare_schedule_with_measurements(schedule_info, instr_to_write_class, measurements)