## The test checks that we can parse and dump a TU index section that is
## compliant to the DWARFv5 standard.

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-tu-index - 2>&1 | \
# RUN:   FileCheck %s --implicit-check-not "could not find unit with signature"

# CHECK:      .debug_tu_index contents:
# CHECK-NEXT: version = 5, units = 1, slots = 2
# CHECK-EMPTY:
# CHECK-NEXT: Index Signature          INFO                                     ABBREV                   LINE                     STR_OFFSETS
# CHECK-NEXT: ----- ------------------ ---------------------------------------- ------------------------ ------------------------ ------------------------
# CHECK-NEXT:     1 0x1100001122222222 [0x0000000000001000, 0x0000000000001010) [0x00002000, 0x00002020) [0x00003000, 0x00003030) [0x00004000, 0x00004040)

    .section .debug_tu_index, "", @progbits
## Header:
    .short 5            # Version
    .space 2            # Padding
    .long 4             # Section count
    .long 1             # Unit count
    .long 2             # Slot count
## Hash Table of Signatures:
    .quad 0x1100001122222222
    .quad 0
## Parallel Table of Indexes:
    .long 1
    .long 0
## Table of Section Offsets:
## Row 0:
    .long 1             # DW_SECT_INFO
    .long 3             # DW_SECT_ABBREV
    .long 4             # DW_SECT_LINE
    .long 6             # DW_SECT_STR_OFFSETS
## Row 1:
    .long 0x1000        # Offset in .debug_info.dwo
    .long 0x2000        # Offset in .debug_abbrev.dwo
    .long 0x3000        # Offset in .debug_line.dwo
    .long 0x4000        # Offset in .debug_str_offsets.dwo
## Table of Section Sizes:
    .long 0x10          # Size in .debug_info.dwo
    .long 0x20          # Size in .debug_abbrev.dwo
    .long 0x30          # Size in .debug_line.dwo
    .long 0x40          # Size in .debug_str_offsets.dwo
