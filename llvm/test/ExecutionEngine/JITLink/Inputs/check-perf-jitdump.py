#!/usr/bin/env python3
"""Check actual perf records and the addresses they describe in perf's ELF."""
import argparse
import pathlib
import struct
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--run", action="store_true")
parser.add_argument("--expect-no-unwind", action="store_true")
parser.add_argument("--no-unwind-for", action="append", default=[])
parser.add_argument("input", nargs=argparse.REMAINDER)
args = parser.parse_args()
run_jit = args.run
if run_jit:
    process = subprocess.Popen(args.input)
    process.wait()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, process.args)
    path = pathlib.Path("/tmp/jit-" + str(process.pid) + ".dump")
else:
    path = pathlib.Path(args.input[0])


def leb(data, offset):
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7


# Read only the dump belonging to the process just launched, so concurrent
# tests cannot consume each other's files in the shared /tmp directory.
data = path.read_bytes()
magic, version, offset, machine = struct.unpack_from("<IIII", data)
assert (magic, version, machine) == (0x4A695444, 1, 62)
assert struct.unpack_from("<Q", data, 32)[0] == 0
pending = None
debug_address = None
functions = []
unwind_count = 0
while offset < len(data):
    kind, size, timestamp = struct.unpack_from("<IIQ", data, offset)
    assert size >= 16 and offset + size <= len(data)
    record = data[offset : offset + size]
    if kind == 4:
        assert pending is None, "Unwind data was not consumed by a load"
        unwind_size, header_size, mapped_size = struct.unpack_from("<QQQ", record, 16)
        assert size == 40 + unwind_size
        assert header_size == 8 and mapped_size == 0
        frame = record[40:-header_size]
        header = record[-header_size:]
        assert header[:4] == b"\x01\x1b\xff\xff"
        assert len(frame) + 4 + struct.unpack_from("<i", header, 4)[0] == 0
        pending = (frame, timestamp)
        unwind_count += 1
    elif kind == 2:
        assert debug_address is None, "Debug data was not consumed by a load"
        debug_address = struct.unpack_from("<Q", record, 16)[0]
    elif kind == 0:
        address, code_size = struct.unpack_from("<QQ", record, 32)
        name = record[56:].split(b"\0", 1)[0].decode()
        assert size == 56 + len(name.encode()) + 1 + code_size
        no_unwind = args.expect_no_unwind or name in args.no_unwind_for
        assert (
            no_unwind or pending is not None
        ), "Every supported load needs its own unwind record"
        assert debug_address is None or debug_address == address
        if no_unwind:
            assert pending is None, "Unsupported CFI must not be emitted"
        else:
            frame, unwind_time = pending
            assert unwind_time <= timestamp
            frame_vaddr = 0x80 + ((code_size + 7) & ~7)
            at, coverage, cies = 0, [], {}
            while at < len(frame):
                length = struct.unpack_from("<I", frame, at)[0]
                if length == 0:
                    assert at + 4 == len(frame), "Bytes follow the frame terminator"
                    break
                assert length >= 4 and at + length + 4 <= len(frame)
                cie = struct.unpack_from("<I", frame, at + 4)[0]
                if not cie:
                    version = frame[at + 8]
                    assert version in (1, 3)
                    end = frame.index(0, at + 9)
                    augmentation = frame[at + 9 : end]
                    assert augmentation == b"zR"
                    pos = end + 1
                    for _ in range(2):
                        _, pos = leb(frame, pos)
                    if version == 1:
                        pos += 1  # Return register is one byte in CIE v1.
                    else:
                        _, pos = leb(frame, pos)
                    aug_size, pos = leb(frame, pos)
                    assert aug_size == 1 and frame[pos] == 0x1B
                    cies[at] = frame[pos]
                else:
                    encoding = cies[at + 4 - cie]
                    assert encoding == 0x1B
                    delta, span = struct.unpack_from("<iI", frame, at + 8)
                    aug_size, _ = leb(frame, at + 16)
                    assert aug_size == 0, "Profiler FDE must not retain an LSDA pointer"
                    pc = frame_vaddr + at + 8 + delta
                    coverage.append((pc, pc + span))
                at += length + 4
            assert (0x80, 0x80 + code_size) in coverage, (name, coverage)
        functions.append((name, no_unwind))
        pending = debug_address = None
    offset += size
assert pending is None and debug_address is None
assert functions
assert unwind_count == sum(not skip for _, skip in functions)
for name, skip in sorted(functions):
    print(
        name
        + (
            ": code load without unwind"
            if skip
            else ": valid per-function unwind record"
        )
    )

if run_jit:
    path.unlink()
