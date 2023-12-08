## This tests handling invalid .debug_str_offsets.dwo sections in
## a pre-standard DWO/DWP file.

# RUN: llvm-mc -triple x86_64 %s -filetype=obj -o %t.dwo
# RUN: not llvm-dwarfdump -v %t.dwo 2>&1 | FileCheck %s

# RUN: llvm-mc -triple x86_64 %s -filetype=obj -o %t.dwp --defsym DWP=0
# RUN: not llvm-dwarfdump -v %t.dwp 2>&1 | FileCheck %s

# CHECK: error: invalid reference to or invalid content in .debug_str_offsets[.dwo]: length exceeds section size

    .section .debug_abbrev.dwo,"e",@progbits
.LAbbr:
    .byte 0x01  # Abbrev code
    .byte 0x11  # DW_TAG_compile_unit
    .byte 0x00  # DW_CHILDREN_no
    .byte 0x00  # EOM(1)
    .byte 0x00  # EOM(2)
    .byte 0x00  # EOM(3)
.LAbbrEnd:

    .section .debug_info.dwo,"e",@progbits
.LCU:
    .long .LCUEnd-.LCUVersion
.LCUVersion:
    .short 4
    .long 0
    .byte 8
    .uleb128 1
.LCUEnd:

## The section is truncated, i.e. its size is not a multiple of entry size.
    .section .debug_str_offsets.dwo,"e",@progbits
.LStrOff:
    .byte 0
.LStrOffEnd:

.ifdef DWP
    .section .debug_cu_index, "", @progbits
## Header:
    .long 2                         # Version
    .long 3                         # Section count
    .long 1                         # Unit count
    .long 2                         # Slot count
## Hash Table of Signatures:
    .quad 0x1100001122222222        # DWO Id of CU0
    .quad 0
## Parallel Table of Indexes:
    .long 1
    .long 0
## Table of Section Offsets:
## Row 0:
    .long 1                         # DW_SECT_INFO
    .long 3                         # DW_SECT_ABBREV
    .long 6                         # DW_SECT_STR_OFFSETS
## Row 1, offsets of the contribution
    .long .LCU-.debug_info.dwo
    .long .LAbbr-.debug_abbrev.dwo
    .long .LStrOff-.debug_str_offsets.dwo
## Table of Section Sizes:
## Row 1, sizes of the contribution
    .long .LCUEnd-.LCU
    .long .LAbbrEnd-.LAbbr
    .long .LStrOffEnd-.LStrOff
.endif
