import struct
import sys
from pathlib import Path


ELF_PROGRAM_HEADER_SIZE = 56
ELF_SECTION_HEADER_SIZE = 64
PT_LOAD = 1
PT_PHDR = 6
SHT_NOBITS = 8


def read_u16(data, offset):
    return struct.unpack_from("<H", data, offset)[0]


def read_u64(data, offset):
    return struct.unpack_from("<Q", data, offset)[0]


def read_program_headers(data):
    offset = read_u64(data, 32)
    entry_size = read_u16(data, 54)
    count = read_u16(data, 56)
    assert entry_size == ELF_PROGRAM_HEADER_SIZE
    end = offset + entry_size * count
    assert end <= len(data), (
        f"program header table [{offset:#x}, {end:#x}) exceeds "
        f"file size {len(data):#x}"
    )
    headers = [
        struct.unpack_from("<IIQQQQQQ", data, offset + i * entry_size)
        for i in range(count)
    ]
    return offset, end, headers


def relocate(path, new_offset):
    path = Path(path)
    data = bytearray(path.read_bytes())
    old_offset = read_u64(data, 32)
    entry_size = read_u16(data, 54)
    count = read_u16(data, 56)
    table = data[old_offset : old_offset + entry_size * count]
    new_end = new_offset + len(table)
    data.extend(bytes(max(0, new_end - len(data))))
    data[new_offset:new_end] = table
    struct.pack_into("<Q", data, 32, new_offset)
    path.write_bytes(data)


def verify(path):
    data = Path(path).read_bytes()
    assert data[:4] == b"\x7fELF"
    assert data[4] == 2, "test helper only supports ELF64"
    assert data[5] == 1, "test helper only supports little-endian ELF"

    phdr_offset, phdr_end, headers = read_program_headers(data)
    table_size = phdr_end - phdr_offset

    phdrs = [header for header in headers if header[0] == PT_PHDR]
    assert len(phdrs) == 1
    assert phdrs[0][2] == phdr_offset
    assert phdrs[0][5] == table_size

    loads = [header for header in headers if header[0] == PT_LOAD]
    assert any(
        header[2] <= phdr_offset and header[2] + header[5] >= phdr_end
        for header in loads
    ), "program header table is not covered by a PT_LOAD"

    section_offset = read_u64(data, 40)
    section_entry_size = read_u16(data, 58)
    section_count = read_u16(data, 60)
    assert section_entry_size == ELF_SECTION_HEADER_SIZE
    section_end = section_offset + section_entry_size * section_count
    assert section_end <= len(data)

    for index in range(section_count):
        header_offset = section_offset + index * section_entry_size
        section_type = struct.unpack_from("<I", data, header_offset + 4)[0]
        file_offset = read_u64(data, header_offset + 24)
        size = read_u64(data, header_offset + 32)
        if section_type != SHT_NOBITS and size:
            assert not (
                file_offset < phdr_end and phdr_offset < file_offset + size
            ), f"section {index} overlaps the program header table"


if sys.argv[1] == "relocate":
    relocate(sys.argv[2], int(sys.argv[3], 0))
elif sys.argv[1] == "verify":
    verify(sys.argv[2])
else:
    raise AssertionError(f"unknown operation: {sys.argv[1]}")
