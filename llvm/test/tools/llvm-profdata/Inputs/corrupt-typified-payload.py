import argparse
import struct

TYPIFIED_PROFILE_SECTION = 33
LEGACY_PROFILE_SECTION = 32
UNKNOWN_PROFILE_TYPE = 127


def read_uleb(data, offset):
    """Decode one ULEB128 value and return it with the following offset."""
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def encode_uleb(value):
    """Encode an integer using canonical ULEB128."""
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        result.append(byte)
        if not value:
            return result


def write_output(data, path):
    """Write the transformed profile to its requested output path."""
    with open(path, "wb") as output_file:
        output_file.write(data)


def read_sections(data):
    """Return the section-count offset and decoded section headers."""
    _, offset = read_uleb(data, 0)
    _, offset = read_uleb(data, offset)
    section_count_offset = offset
    section_count = struct.unpack_from("<Q", data, offset)[0]
    offset += 8

    sections = []
    for _ in range(section_count):
        section_header_offset = offset
        section_type, _, section_offset, section_size = struct.unpack_from(
            "<QQQQ", data, offset
        )
        offset += 32
        sections.append(
            (section_header_offset, section_type, section_offset, section_size)
        )
    return section_count_offset, sections


parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--donor")
parser.add_argument(
    "modification",
    choices=(
        "undersized",
        "oversized",
        "unknown",
        "unknown-out-of-bounds",
        "prepended-unknown",
        "duplicate-type",
        "empty-section",
        "prepend-empty-typified",
        "prepend-typified",
    ),
)
args = parser.parse_args()

with open(args.input, "rb") as input_file:
    data = bytearray(input_file.read())

section_count_offset, sections = read_sections(data)
section_count = len(sections)
profile_offset = None
profile_size = None
profile_header_offset = None
for section_header_offset, section_type, section_offset, section_size in sections:
    if section_type == TYPIFIED_PROFILE_SECTION:
        assert profile_offset is None, "expected exactly one typified profile section"
        profile_offset = section_offset
        profile_size = section_size
        profile_header_offset = section_header_offset

if args.modification in ("prepend-empty-typified", "prepend-typified"):
    legacy_sections = [
        section for section in sections if section[1] == LEGACY_PROFILE_SECTION
    ]
    assert legacy_sections, "expected a legacy profile section"
    legacy_header_offset, _, legacy_offset, _ = legacy_sections[0]
    header_size = struct.calcsize("<QQQQ")

    # Growing the section table shifts every section payload by one header.
    for section_header_offset, _, section_offset, _ in sections:
        struct.pack_into(
            "<Q", data, section_header_offset + 16, section_offset + header_size
        )

    typified_payload = b""
    typified_offset = legacy_offset + header_size
    if args.modification == "prepend-typified":
        assert args.donor, "expected --donor for a nonempty typified section"
        with open(args.donor, "rb") as donor_file:
            donor_data = donor_file.read()
        _, donor_sections = read_sections(donor_data)
        donor_profiles = [
            section
            for section in donor_sections
            if section[1] == TYPIFIED_PROFILE_SECTION and section[3] != 0
        ]
        assert len(donor_profiles) == 1, "expected one nonempty donor section"
        _, _, donor_offset, donor_size = donor_profiles[0]
        typified_payload = donor_data[donor_offset : donor_offset + donor_size]
        typified_offset = len(data) + header_size

    # Place the typified section immediately before a valid legacy one to
    # verify that its decoding mode does not leak into the following section.
    typified_header = struct.pack(
        "<QQQQ",
        TYPIFIED_PROFILE_SECTION,
        0,
        typified_offset,
        len(typified_payload),
    )
    data[legacy_header_offset:legacy_header_offset] = typified_header
    data.extend(typified_payload)
    struct.pack_into("<Q", data, section_count_offset, section_count + 1)
    write_output(data, args.output)
    raise SystemExit

assert profile_offset is not None
assert profile_size is not None
assert profile_header_offset is not None
profile_start = profile_offset
profile_end = profile_offset + profile_size
_, profile_offset = read_uleb(data, profile_offset)
profile_count_offset = profile_offset
profile_count, type_offset = read_uleb(data, profile_offset)
assert profile_count == 1
assert type_offset == profile_count_offset + 1
profile_type, size_offset = read_uleb(data, type_offset)
assert profile_type == 0
payload_size, payload_offset = read_uleb(data, size_offset)
assert payload_offset + payload_size < profile_end


def adjust_sections_after(insertion_offset, size_delta):
    """Adjust section offsets and typified section size after a byte edit."""
    if not size_delta:
        return
    for section_header_offset, _, section_offset, _ in sections:
        if section_offset > insertion_offset:
            struct.pack_into(
                "<Q",
                data,
                section_header_offset + 16,
                section_offset + size_delta,
            )
    struct.pack_into(
        "<Q",
        data,
        profile_header_offset + 24,
        profile_size + size_delta,
    )


def replace_payload_size(new_size):
    """Replace the ULEB128 payload size and repair affected section metadata."""
    encoded_size = encode_uleb(new_size)
    old_encoded_size = payload_offset - size_offset
    data[size_offset:payload_offset] = encoded_size
    adjust_sections_after(size_offset, len(encoded_size) - old_encoded_size)


if args.modification == "undersized":
    assert payload_size > 0
    replace_payload_size(payload_size - 1)
elif args.modification == "oversized":
    replace_payload_size(payload_size + 1)
elif args.modification == "unknown":
    data[type_offset] = UNKNOWN_PROFILE_TYPE
elif args.modification == "unknown-out-of-bounds":
    data[type_offset] = UNKNOWN_PROFILE_TYPE
    replace_payload_size((1 << 64) - 1)
elif args.modification == "prepended-unknown":
    unknown_payload = b"\xa5"
    unknown_block = (
        bytes([UNKNOWN_PROFILE_TYPE])
        + encode_uleb(len(unknown_payload))
        + unknown_payload
    )
    insertion_offset = type_offset
    data[profile_count_offset] = 2
    data[insertion_offset:insertion_offset] = unknown_block
    adjust_sections_after(insertion_offset, len(unknown_block))
elif args.modification == "duplicate-type":
    block_end = payload_offset + payload_size
    duplicate_block = bytes(data[type_offset:block_end])
    data[profile_count_offset] = 2
    data[block_end:block_end] = duplicate_block
    adjust_sections_after(block_end, len(duplicate_block))
else:
    del data[profile_start:profile_end]
    for section_header_offset, _, section_offset, _ in sections:
        if section_offset > profile_start:
            struct.pack_into(
                "<Q",
                data,
                section_header_offset + 16,
                section_offset - profile_size,
            )
    struct.pack_into("<Q", data, profile_header_offset + 24, 0)

write_output(data, args.output)
