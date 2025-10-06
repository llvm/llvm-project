#!/usr/bin/env python3
"""
Parse llvm-exegesis measurements and compare with X86ScheduleZnver5.td definitions.
"""

import yaml
import re
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

def parse_tablegen(schedule_td_file: str = "llvm/lib/Target/X86/X86ScheduleZnver5.td") -> Tuple[Dict, Dict]:
    """Parse X86ScheduleZnver5.td to extract scheduling definitions"""
    print(f"\nParsing {schedule_td_file}...")

    schedule_info = {}
    instr_to_write_class = {}

    with open(schedule_td_file, 'r') as f:
        content = f.read()

    # Parse Write class definitions from Zn5WriteResIntPair/FPPair
    # Example: defm : Zn5WriteResIntPair<WriteIMul32Reg, [Zn5Multiplier012], 3, [1], 1>;
    pattern1 = r'defm?\s+:\s+Zn5WriteRes(?:Int|FP)?Pair<(Write\w+),\s*\[([^\]]+)\],\s*(\d+),\s*\[([^\]]+)\],\s*(\d+)'

    for match in re.finditer(pattern1, content):
        write_class = match.group(1)
        resources = match.group(2)
        latency = int(match.group(3))
        release_cycles = match.group(4)
        num_uops = int(match.group(5))

        schedule_info[write_class] = {
            'resources': resources,
            'latency': latency,
            'release_cycles': release_cycles,
            'num_uops': num_uops,
            'type': 'pair'
        }

    # Parse direct SchedWriteRes definitions
    # Example: def Zn5WriteCMPXCHG8rr : SchedWriteRes<[Zn5ALU012345]> {
    pattern2 = r'def\s+(Zn5Write\w+)\s*:\s*SchedWriteRes<\[([^\]]+)\]>\s*\{([^}]+)\}'

    for match in re.finditer(pattern2, content, re.DOTALL):
        zn5_write = match.group(1)
        resources = match.group(2)
        body = match.group(3)

        # Extract fields from body
        latency_match = re.search(r'Latency\s*=\s*(\d+)', body)
        latency = int(latency_match.group(1)) if latency_match else 1

        release_match = re.search(r'ReleaseAtCycles\s*=\s*\[([^\]]+)\]', body)
        release_cycles = release_match.group(1) if release_match else str(latency)

        uops_match = re.search(r'NumMicroOps\s*=\s*(\d+)', body)
        num_uops = int(uops_match.group(1)) if uops_match else 1

        schedule_info[zn5_write] = {
            'resources': resources,
            'latency': latency,
            'release_cycles': release_cycles,
            'num_uops': num_uops,
            'type': 'direct'
        }

    # Parse InstRW mappings to find which instructions use which scheduling
    # Example: def : InstRW<[Zn5WriteALUSlow], (instrs ADD8i8, ADD16i16)>;
    pattern3 = r'def\s*:\s*InstRW<\[([\w\s,]+)\],\s*\(instrs\s+([^)]+)\)>'

    for match in re.finditer(pattern3, content):
        sched_writes = match.group(1).strip()
        instrs_str = match.group(2)

        # Parse instruction list - handle both comma and whitespace separation
        instrs = [i.strip() for i in re.split(r'[,\s]+', instrs_str) if i.strip()]

        for instr in instrs:
            # Take the first scheduling write if multiple
            write_class = sched_writes.split(',')[0].strip()
            instr_to_write_class[instr] = write_class

    print(f"Parsed {len(schedule_info)} scheduling definitions")
    print(f"Parsed {len(instr_to_write_class)} instruction-to-write-class mappings")

    return schedule_info, instr_to_write_class

def compare_schedule_with_measurements(schedule_info: Dict, instr_to_write_class: Dict, measurements: Dict):
    """Compare schedule definitions with actual measurements"""
    print("\n=== Comparing Schedule Info with Measurements ===\n")

    # Create reverse mapping: Write class -> list of instructions
    write_class_to_instrs = {}
    for instr, write_class in instr_to_write_class.items():
        if write_class not in write_class_to_instrs:
            write_class_to_instrs[write_class] = []
        write_class_to_instrs[write_class].append(instr)

    # For each schedule definition, find corresponding measurements
    for write_class, sched_def in schedule_info.items():
        # Find instructions that use this Write class
        instrs_using_write = []

        # Direct match in Write classes
        if write_class in write_class_to_instrs:
            instrs_using_write = write_class_to_instrs[write_class]

        # Also check for Zn5Write* classes (custom Zen5 definitions)
        for instr, wc in instr_to_write_class.items():
            if wc == write_class or write_class in wc:
                if instr not in instrs_using_write:
                    instrs_using_write.append(instr)

        # Find measurements for these instructions
        measured_instrs = []
        for instr in instrs_using_write:
            if instr in measurements:
                measured_instrs.append((instr, measurements[instr]))

        if measured_instrs:
            print(f"{write_class}:")

            # Calculate expected throughput from release cycles
            release_str = sched_def['release_cycles']
            if release_str.isdigit():
                expected_throughput = int(release_str)
            else:
                # Parse list like "1, 1, 2" and take the max
                cycles = [int(x.strip()) for x in release_str.split(',') if x.strip().isdigit()]
                expected_throughput = max(cycles) if cycles else 1

            print(f"  Schedule: Latency={sched_def['latency']}, Resources={sched_def['resources']}, Release={sched_def['release_cycles']}, Throughput≈{expected_throughput}")
            print(f"  Found {len(measured_instrs)} measured instructions:")

            for instr, meas in measured_instrs:
                output = f"    {instr}:"
                if 'latency' in meas:
                    lat_diff = abs(meas['latency'] - sched_def['latency'])
                    status = "✓" if lat_diff <= 0.5 else "⚠"
                    output += f" latency={meas['latency']:.4f} (expected={sched_def['latency']}) {status}"
                if 'inverse_throughput' in meas:
                    thr_diff = abs(meas['inverse_throughput'] - expected_throughput)
                    status = "✓" if thr_diff/expected_throughput <= 0.2 else "⚠"
                    output += f" throughput={meas['inverse_throughput']:.4f} (expected≈{expected_throughput}) {status}"
                print(output)

            print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        measurements_file = sys.argv[1]
    else:
        measurements_file = "znver5_rthroughput.yaml"
        # measurements_file = "znver5_latency.yaml"

    measurements = parse_measurements_test(measurements_file)
    schedule_info, instr_to_write_class = parse_tablegen()

    # Compare schedule definitions with measurements
    if measurements and schedule_info:
        compare_schedule_with_measurements(schedule_info, instr_to_write_class, measurements)