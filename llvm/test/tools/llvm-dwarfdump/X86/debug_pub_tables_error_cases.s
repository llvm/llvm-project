# RUN: llvm-mc -triple x86_64 %s -filetype=obj -o %t

## All four name lookup table sections share the same parser, but slightly
## different code paths are used to reach it. Do a comprehensive check for one
## of the sections and minimal checks for the others.

# RUN: not llvm-dwarfdump -debug-gnu-pubnames %t 2> %t.err | FileCheck %s
# RUN: FileCheck %s --input-file=%t.err --check-prefix=ERR

# RUN: not llvm-dwarfdump -debug-pubnames -debug-pubtypes -debug-gnu-pubtypes %t 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR-MIN

    .section .debug_gnu_pubnames,"",@progbits
# CHECK:      .debug_gnu_pubnames contents:

## The next few sets do not contain all required fields in the header.
# ERR: error: name lookup table at offset 0x0 does not have a complete header: unexpected end of data at offset 0x5 while reading [0x4, 0x6)
# CHECK-NEXT: length = 0x00000001, format = DWARF32, version = 0x0000, unit_offset = 0x00000000, unit_size = 0x00000000
# CHECK-NEXT: Offset Linkage Kind Name
# CHECK-NOT:  0x
    .long .LSet0End-.LSet0      # Length
.LSet0:
    .byte 1                     # Version (truncated)
.LSet0End:

# ERR: error: name lookup table at offset 0x5 does not have a complete header: unexpected end of data at offset 0xe while reading [0xb, 0xf)
# CHECK-NEXT: length = 0x00000005, format = DWARF32, version = 0x0002, unit_offset = 0x00000000, unit_size = 0x00000000
# CHECK-NEXT: Offset Linkage Kind Name
# CHECK-NOT:  0x
    .long .LSet1End-.LSet1      # Length
.LSet1:
    .short 2                    # Version
    .byte 1, 2, 3               # Debug Info Offset (truncated)
.LSet1End:

# ERR: error: name lookup table at offset 0xe does not have a complete header: unexpected end of data at offset 0x1b while reading [0x18, 0x1c)
# CHECK-NEXT: length = 0x00000009, format = DWARF32, version = 0x0002, unit_offset = 0x00000032, unit_size = 0x00000000
# CHECK-NEXT: Offset Linkage Kind Name
# CHECK-NOT:  0x
    .long .LSet2End-.LSet2      # Length
.LSet2:
    .short 2                    # Version
    .long 0x32                  # Debug Info Offset
    .byte 1, 2, 3               # Debug Info Length (truncated)
.LSet2End:

## This set is terminated just after the header.
# ERR: error: name lookup table at offset 0x1b parsing failed: unexpected end of data at offset 0x29 while reading [0x29, 0x2d)
# CHECK-NEXT: length = 0x0000000a, format = DWARF32, version = 0x0002, unit_offset = 0x00000048, unit_size = 0x00000064
# CHECK-NEXT: Offset Linkage Kind Name
# CHECK-NOT:  0x
    .long .LSet3End-.LSet3      # Length
.LSet3:
    .short 2                    # Version
    .long 0x48                  # Debug Info Offset
    .long 0x64                  # Debug Info Length
.LSet3End:

## The offset in the first pair is truncated.
# ERR: error: name lookup table at offset 0x29 parsing failed: unexpected end of data at offset 0x3a while reading [0x37, 0x3b)
# CHECK-NEXT: length = 0x0000000d, format = DWARF32, version = 0x0002, unit_offset = 0x000000ac, unit_size = 0x00000036
# CHECK-NEXT: Offset Linkage Kind Name
# CHECK-NOT:  0x
    .long .LSet4End-.LSet4      # Length
.LSet4:
    .short 2                    # Version
    .long 0xac                  # Debug Info Offset
    .long 0x36                  # Debug Info Length
    .byte 1, 2, 3               # Offset (truncated)
.LSet4End:

## The set is truncated just after the offset of the first pair.
# ERR: error: name lookup table at offset 0x3a parsing failed: unexpected end of data at offset 0x4c while reading [0x4c, 0x4d)
# CHECK-NEXT: length = 0x0000000e, format = DWARF32, version = 0x0002, unit_offset = 0x000000e2, unit_size = 0x00000015
# CHECK-NEXT: Offset Linkage Kind Name
# CHECK-NOT:  0x
    .long .LSet5End-.LSet5      # Length
.LSet5:
    .short 2                    # Version
    .long 0xe2                  # Debug Info Offset
    .long 0x15                  # Debug Info Length
    .long 0xf4                  # Offset
.LSet5End:

## The set is truncated just after the index entry field of the first pair.
# ERR: error: name lookup table at offset 0x4c parsing failed: no null terminated string at offset 0x5f
# CHECK-NEXT: length = 0x0000000f, format = DWARF32, version = 0x0002, unit_offset = 0x000000f7, unit_size = 0x00000010
# CHECK-NEXT: Offset Linkage Kind Name
# CHECK-NOT:  0x
    .long .LSet6End-.LSet6      # Length
.LSet6:
    .short 2                    # Version
    .long 0xf7                  # Debug Info Offset
    .long 0x10                  # Debug Info Length
    .long 0xf4                  # Offset
    .byte 0x30                  # Index Entry
.LSet6End:

## This set contains a string which is not properly terminated.
# ERR: error: name lookup table at offset 0x5f parsing failed: no null terminated string at offset 0x72
# CHECK-NEXT: length = 0x00000012, format = DWARF32, version = 0x0002, unit_offset = 0x00000107, unit_size = 0x0000004b
# CHECK-NEXT: Offset Linkage Kind Name
# CHECK-NOT:  0x
    .long .LSet7End-.LSet7      # Length
.LSet7:
    .short 2                    # Version
    .long 0x107                 # Debug Info Offset
    .long 0x4b                  # Debug Info Length
    .long 0x111                 # Offset
    .byte 0x30                  # Index Entry
    .ascii "foo"                # The string does not terminate before the set data ends.
.LSet7End:

## This set occupies some space after the terminator.
# ERR: error: name lookup table at offset 0x75 has a terminator at offset 0x8c before the expected end at 0x8d
# CHECK-NEXT: length = 0x00000018, format = DWARF32, version = 0x0002, unit_offset = 0x00000154, unit_size = 0x000002ac
# CHECK-NEXT: Offset Linkage Kind Name
# CHECK-NEXT: 0x0000018e EXTERNAL FUNCTION "foo"
# CHECK-NOT:  0x
    .long .LSet8End-.LSet8      # Length
.LSet8:
    .short 2                    # Version
    .long 0x154                 # Debug Info Offset
    .long 0x2ac                 # Debug Info Length
    .long 0x18e                 # Offset
    .byte 0x30                  # Index Entry
    .asciz "foo"                # Name
    .long 0                     # Terminator
    .space 1
.LSet8End:

## The remaining space in the section is too short to even contain a unit length
## field.
# ERR: error: name lookup table at offset 0x91 parsing failed: unexpected end of data at offset 0x94 while reading [0x91, 0x95)
# CHECK-NOT:  length =
    .space 3

# ERR-MIN:      .debug_pubnames contents:
# ERR-MIN-NEXT: error: name lookup table at offset 0x0 parsing failed: unexpected end of data at offset 0x1 while reading [0x0, 0x4)
# ERR-MIN:      .debug_pubtypes contents:
# ERR-MIN-NEXT: error: name lookup table at offset 0x0 parsing failed: unexpected end of data at offset 0x1 while reading [0x0, 0x4)
# ERR-MIN:      .debug_gnu_pubtypes contents:
# ERR-MIN-NEXT: error: name lookup table at offset 0x0 parsing failed: unexpected end of data at offset 0x1 while reading [0x0, 0x4)

    .section .debug_pubnames,"",@progbits
    .byte 0
    .section .debug_pubtypes,"",@progbits
    .byte 0
    .section .debug_gnu_pubtypes,"",@progbits
    .byte 0
