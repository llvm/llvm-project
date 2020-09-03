## The test checks that we can parse and dump a pre-standard TU index section.
## See https://gcc.gnu.org/wiki/DebugFissionDWP for the proposal.

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-tu-index - | \
# RUN:   FileCheck %s

# CHECK:      .debug_tu_index contents:
# CHECK-NEXT: version = 2, units = 1, slots = 2
# CHECK-EMPTY:
# CHECK-NEXT: Index Signature          TYPES                    ABBREV                   LINE                     STR_OFFSETS
# CHECK-NEXT: ----- ------------------ ------------------------ ------------------------ ------------------------ ------------------------
# CHECK-NEXT:     1 0x1100001122222222 [0x00001000, 0x00001010) [0x00002000, 0x00002020) [0x00003000, 0x00003030) [0x00004000, 0x00004040)

    .section .debug_tu_index, "", @progbits
## Header:
    .long 2             # Version
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
    .long 2             # DW_SECT_TYPES
    .long 3             # DW_SECT_ABBREV
    .long 4             # DW_SECT_LINE
    .long 6             # DW_SECT_STR_OFFSETS
## Row 1:
    .long 0x1000        # Offset in .debug_types.dwo
    .long 0x2000        # Offset in .debug_abbrev.dwo
    .long 0x3000        # Offset in .debug_line.dwo
    .long 0x4000        # Offset in .debug_str_offsets.dwo
## Table of Section Sizes:
    .long 0x10          # Size in .debug_types.dwo
    .long 0x20          # Size in .debug_abbrev.dwo
    .long 0x30          # Size in .debug_line.dwo
    .long 0x40          # Size in .debug_str_offsets.dwo
