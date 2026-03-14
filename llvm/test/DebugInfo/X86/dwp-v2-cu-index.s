## The test checks that we can parse and dump a pre-standard CU index section.
## See https://gcc.gnu.org/wiki/DebugFissionDWP for the proposal.

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-cu-index - | \
# RUN:   FileCheck %s

# CHECK:      .debug_cu_index contents:
# CHECK-NEXT: version = 2, units = 1, slots = 2
# CHECK-EMPTY:
# CHECK-NEXT: Index Signature          INFO                                     ABBREV                   LINE                     LOC                      STR_OFFSETS              MACINFO                  MACRO
# CHECK-NEXT: ----- ------------------ ---------------------------------------- ------------------------ ------------------------ ------------------------ ------------------------ ------------------------ ------------------------
# CHECK-NEXT:     1 0x1100001122222222 [0x0000000000001000, 0x0000000000001010) [0x00002000, 0x00002020) [0x00003000, 0x00003030) [0x00004000, 0x00004040) [0x00005000, 0x00005050) [0x00006000, 0x00006060) [0x00007000, 0x00007070)

    .section .debug_cu_index, "", @progbits
## Header:
    .long 2             # Version
    .long 7             # Section count
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
    .long 5             # DW_SECT_LOC
    .long 6             # DW_SECT_STR_OFFSETS
    .long 7             # DW_SECT_MACINFO
    .long 8             # DW_SECT_MACRO
## Row 1:
    .long 0x1000        # Offset in .debug_info.dwo
    .long 0x2000        # Offset in .debug_abbrev.dwo
    .long 0x3000        # Offset in .debug_line.dwo
    .long 0x4000        # Offset in .debug_loc.dwo
    .long 0x5000        # Offset in .debug_str_offsets.dwo
    .long 0x6000        # Offset in .debug_macinfo.dwo
    .long 0x7000        # Offset in .debug_macro.dwo
## Table of Section Sizes:
    .long 0x10          # Size in .debug_info.dwo
    .long 0x20          # Size in .debug_abbrev.dwo
    .long 0x30          # Size in .debug_line.dwo
    .long 0x40          # Size in .debug_loc.dwo
    .long 0x50          # Size in .debug_str_offsets.dwo
    .long 0x60          # Size in .debug_macinfo.dwo
    .long 0x70          # Size in .debug_macro.dwo
