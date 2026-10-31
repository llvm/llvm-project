#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check RV64 libomp microtask unwind rules at instruction boundaries.

The bounded model assumes nonnegative argc, disjoint argument/OMPT buffers,
and an ABI-conforming microtask that preserves sp/s0 and the saved slots.
It tracks sp, s0, ra and the two saved register slots. Unspecified initial
register rules follow the GNU libgcc unsaved-register convention.
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys


def require(condition, message=None):
    """Validate input/model contracts even when Python optimization is enabled."""
    if not condition:
        raise AssertionError(message)


def add(value, amount):
    return (value[0], value[1] + amount)

def inspect(disassembly, frames):
    instructions = [(int(pc, 16), asm.strip()) for pc, asm in re.findall(
        r'^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(.+)$', disassembly, re.M)]
    require(instructions and instructions[0][0] == 0)
    # Require the reviewed sole-function object and the two-slot RV64 frame.
    require(len(re.findall(r'FDE cie=', frames)) == 1)
    fde = frames.split('FDE cie=', 1)[1]
    rows = [(int(pc, 16), cfa, ra, fp) for pc, cfa, ra, fp in re.findall(
        r'^([0-9a-f]{16})\s+(\S+)\s+(\S+)\s+(\S+)\s*$', fde, re.M)]
    require(rows and rows[0][0] == 0)
    require(all(pc in {x[0] for x in instructions} for pc, *_ in rows))
    pcs = [row[0] for row in rows]
    registers = {'sp': ('entry-sp', 0), 's0': ('caller-fp', 0), 'ra': ('caller-ra', 0)}
    memory = {}
    transitions = []
    checked = []
    for pc, asm in instructions:
        _, cfa_rule, ra_rule, fp_rule = rows[bisect.bisect_right(pcs, pc) - 1]
        cfa_match = re.fullmatch(r'(sp|s0)([+-]\d+)', cfa_rule)
        require(cfa_match, cfa_rule)
        cfa = add(registers[cfa_match[1]], int(cfa_match[2]))

        def recover(rule, register):
            # The reviewed CIE has no explicit undefined rule. GNU libgcc uses
            # its initial unsaved-register rule, including after cfi_restore.
            if rule == 'u':
                return registers[register]
            match = re.fullmatch(r'c(-?\d+)', rule)
            require(match, rule)
            return memory.get(add(cfa, int(match[1])), ('unavailable', 0))

        errors = []
        if cfa != ('entry-sp', 0):
            errors.append('CFA')
        if recover(ra_rule, 'ra') != ('caller-ra', 0):
            errors.append('ra')
        if recover(fp_rule, 's0') != ('caller-fp', 0):
            errors.append('s0')
        checked.append(dict(pc=hex(pc), instruction=asm, cfa=cfa_rule,
                            ra=ra_rule, s0=fp_rule, errors=errors))
        operation = re.sub(r'\s+', ' ', asm).replace('fp', 's0')
        changed = True
        match = re.fullmatch(r'addi (sp|s0),(sp|s0),(-?\d+)', operation)
        if match:
            registers[match[1]] = add(registers[match[2]], int(match[3]))
        elif operation in ('sd ra,8(sp)', 'sd s0,0(sp)'):
            register, offset = ('ra', 8) if operation.startswith('sd ra') else ('s0', 0)
            memory[add(registers['sp'], offset)] = registers[register]
        elif operation in ('ld ra,8(sp)', 'ld s0,0(sp)'):
            register, offset = ('ra', 8) if operation.startswith('ld ra') else ('s0', 0)
            registers[register] = memory[add(registers['sp'], offset)]
        elif operation == 'sub sp,sp,t0':
            registers['sp'] = ('dynamic-sp', 0)
        elif operation == 'andi sp,sp,-16':
            registers['sp'] = ('aligned-dynamic-sp', 0)
        elif operation == 'jalr t0':
            registers['ra'] = ('microtask-return', 0)
        else:
            # Fail closed if a different compiler/primitive introduces another
            # write to a tracked register. Source/CFI changes need model review.
            if re.match(r's[bdwh] ', operation):
                require(re.fullmatch(r'(?:sd s0,0\(a5\)|sw a[12],-2[04]\(s0\)|sd t2,0\(t1\))', operation), operation)
            else:
                require(not re.match(r'\w+ (?:sp|s0|ra),', operation), operation)
            require(not re.match(r'(?:jal|jalr|call|tail)\b', operation), operation)
            changed = False
        if changed:
            transitions.append(dict(pc=hex(pc), instruction=operation))
    require(registers == {'sp': ('entry-sp', 0), 's0': ('caller-fp', 0), 'ra': ('caller-ra', 0)})
    require(len(transitions) == 11, transitions)
    return dict(instruction_boundaries=len(checked), transitions=transitions, rows=checked,
                failures=[row for row in checked if row['errors']])

class Unsupported(Exception):
    """An input or host tool is outside this test's declared capabilities."""


def command(argv):
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode:
        raise RuntimeError('Command failed: ' + repr(argv) + '\n' + result.stderr)
    return result.stdout


def symbol_location(symbols):
    matches = set()
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[3] == 'FUNC' and fields[6] != 'UND' and fields[7].split('@')[0] == '__kmp_invoke_microtask':
            matches.add((int(fields[1], 16), int(fields[2])))
    if len(matches) != 1 or next(iter(matches))[1] <= 0:
        raise Unsupported('Need one defined __kmp_invoke_microtask function symbol')
    return next(iter(matches))


def admission(library, objdump, readelf):
    raw = library.read_bytes()
    if len(raw) < 64 or raw[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', raw, 18)[0] != 243:
        raise Unsupported('Need a little-endian RV64 ELF input')
    versions = {}
    for name, executable in (('objdump', objdump), ('readelf', readelf)):
        if not executable:
            raise Unsupported('GNU ' + name + ' unavailable')
        try:
            version = command([executable, '--version']).splitlines()[0]
        except (OSError, RuntimeError) as exc:
            raise Unsupported('GNU ' + name + ' unavailable: ' + str(exc)) from exc
        if 'GNU' not in version:
            raise Unsupported('GNU ' + name + ' output required; found ' + version)
        versions[name] = dict(path=executable, version=version)
    if 'elf64-littleriscv' not in command([objdump, '-i']):
        raise Unsupported('objdump lacks RV64 support')
    address, size = symbol_location(command([readelf, '-Ws', str(library)]))
    return dict(address=address, size=size, tools=versions, sha256=hashlib.sha256(raw).hexdigest())


def select_frame(frames, raw_frames, address, size):
    # Match the linked function's exact FDE; unrelated library frames are ignored.
    blocks = re.split(r'\n\s*\n', frames)
    candidates = []
    for block in blocks:
        header = re.search(r'^([0-9a-f]+)\s+.*FDE cie=([0-9a-f]+) pc=([0-9a-f]+)\.\.([0-9a-f]+)', block, re.M)
        if header and int(header[3], 16) == address and int(header[4], 16) == address + size:
            candidates.append((header, block))
    require(len(candidates) == 1, 'Expected exactly one complete microtask FDE')
    header, block = candidates[0]
    # Distinguish explicitly undefined DWARF rules from GNU's initial unsaved rule.
    for offset in (header[1], header[2]):
        raw_block = re.search(r'^0*' + re.escape(offset.lstrip('0') or '0') + r'\s+.*?(?=\n\s*\n|\Z)', raw_frames, re.M | re.S)
        require(raw_block, 'Cannot locate referenced CIE/FDE operations')
        require('DW_CFA_undefined' not in raw_block[0], 'Explicit undefined register rule requires model review')
    normalized = []
    for line in block.splitlines():
        row = re.match(r'^([0-9a-f]{16})(\s+.*)$', line)
        if row:
            pc = int(row[1], 16)
            require(address <= pc < address + size)
            normalized.append(format(pc - address, '016x') + row[2])
    require(normalized and int(normalized[0].split()[0], 16) == 0, 'Missing entry unwind state')
    return 'FDE cie=00000000\n' + '\n'.join(normalized) + '\n'


def normalize_instructions(disassembly, address, size):
    rows = []
    for line in disassembly.splitlines():
        match = re.match(r'^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(.+)$', line)
        if match:
            pc = int(match[1], 16)
            require(address <= pc < address + size, 'Instruction outside requested symbol')
            rows.append(format(pc-address, 'x') + ': ' + match[2] + ' ' + match[3])
    require(rows, 'Empty microtask disassembly')
    return '\n'.join(rows)


def check(library, objdump, readelf, admitted):
    address, size = admitted['address'], admitted['size']
    disassembly = command([objdump, '-d', '--disassemble=__kmp_invoke_microtask', str(library)])
    frames = command([readelf, '--debug-dump=frames-interp', str(library)])
    raw_frames = command([readelf, '--debug-dump=frames', str(library)])
    normalized_disassembly = normalize_instructions(disassembly, address, size)
    normalized_frames = select_frame(frames, raw_frames, address, size)
    result = inspect(normalized_disassembly, normalized_frames)
    result.update(admitted, library=str(library), symbol='__kmp_invoke_microtask')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--objdump', required=True)
    parser.add_argument('--readelf', required=True)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    result = dict(status='started', library=str(args.library))
    exit_code = 1
    try:
        admitted = admission(args.library, args.objdump, args.readelf)
        if args.probe:
            result.update(admitted, status='supported')
            exit_code = 0
        else:
            result = check(args.library, args.objdump, args.readelf, admitted)
            result['status'] = 'PASS' if not result['failures'] else 'FAIL'
            exit_code = int(bool(result['failures']))
            print('RV64_CFI boundaries=' + str(result['instruction_boundaries']) + ' errors=' + str(len(result['failures'])))
            for row in result['failures']:
                print(row['pc'] + ': ' + ','.join(row['errors']))
    except Unsupported as exc:
        result.update(status='UNSUPPORTED', reason=str(exc))
        print('RV64_CFI unsupported: ' + str(exc))
        exit_code = 77
    except (AssertionError, OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
        result.update(status='FAIL', error=str(exc))
        print('RV64_CFI failure: ' + str(exc))
    finally:
        if args.json:
            args.json.write_text(json.dumps(result, indent=2) + '\n')
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
