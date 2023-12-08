## This tests dumping a .debug_str_offsets.dwo section in a DWP file when it is
## referenced by units in different formats: one unit is DWARF32 and another
## is DWARF64, thus the .debug_str_offsets.dwo section has contributions with
## different sizes of offsets.
## This also checks that attributes in the units which use the DW_FORM_strx form
## are dumped correctly.

# RUN: llvm-mc -triple x86_64 %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -v - | \
# RUN:   FileCheck %s

# CHECK:      .debug_info.dwo contents:
# CHECK:      Compile Unit:
# CHECK:      DW_TAG_compile_unit [1]
# CHECK-NEXT:   DW_AT_producer [DW_FORM_strx] (indexed (00000000) string = "CU0 Producer")
# CHECK-NEXT:   DW_AT_name [DW_FORM_strx] (indexed (00000001) string = "CU0 Name")
# CHECK:      Compile Unit:
# CHECK:      DW_TAG_compile_unit [1]
# CHECK-NEXT:   DW_AT_producer [DW_FORM_strx] (indexed (00000000) string = "CU1 Producer")
# CHECK-NEXT:   DW_AT_name [DW_FORM_strx] (indexed (00000001) string = "CU1 Name")

# CHECK:      .debug_str.dwo contents:
# CHECK-NEXT: 0x00000000: "CU0 Producer"
# CHECK-NEXT: 0x0000000d: "CU0 Name"
# CHECK-NEXT: 0x00000016: "CU1 Producer"
# CHECK-NEXT: 0x00000023: "CU1 Name"

# CHECK:      .debug_str_offsets.dwo contents:
# CHECK-NEXT: 0x00000000: Contribution size = 8, Format = DWARF32, Version = 4
# CHECK-NEXT: 0x00000000: 00000000 "CU0 Producer"
# CHECK-NEXT: 0x00000004: 0000000d "CU0 Name"
# CHECK-NEXT: 0x00000008: Contribution size = 16, Format = DWARF64, Version = 4
# CHECK-NEXT: 0x00000008: 0000000000000016 "CU1 Producer"
# CHECK-NEXT: 0x00000010: 0000000000000023 "CU1 Name"

    .section .debug_str.dwo, "MSe", @progbits, 1
.LStr0:
    .asciz "CU0 Producer"
.LStr1:
    .asciz "CU0 Name"
.LStr2:
    .asciz "CU1 Producer"
.LStr3:
    .asciz "CU1 Name"

    .section .debug_str_offsets.dwo, "e", @progbits
## The contribution of CU0 (DWARF32)
.LSO0:
    .long .LStr0-.debug_str.dwo     # 0: "CU0 Producer"
    .long .LStr1-.debug_str.dwo     # 1: "CU0 Name"
.LSO0End:
## The contribution of CU1 (DWARF64)
.LSO1:
    .quad .LStr2-.debug_str.dwo     # 0: "CU1 Producer"
    .quad .LStr3-.debug_str.dwo     # 1: "CU1 Name"
.LSO1End:

    .section .debug_abbrev.dwo, "e", @progbits
## For simplicity and to make the test shorter, both compilation units share
## the same abbreviations table.
.LAbbr:
    .uleb128 0x01                   # Abbrev code
    .uleb128 0x11                   # DW_TAG_compile_unit
    .byte 0x00                      # DW_CHILDREN_no
    .uleb128 0x25                   # DW_AT_producer
    .uleb128 0x1a                   # DW_FORM_strx
    .uleb128 0x03                   # DW_AT_name
    .uleb128 0x1a                   # DW_FORM_strx
    .uleb128 0x2131                 # DW_AT_GNU_dwo_id
    .uleb128 0x07                   # DW_FORM_data8
    .byte 0x00                      # EOM(1)
    .byte 0x00                      # EOM(2)
    .byte 0x00                      # EOM(3)
.LAbbrEnd:

    .section .debug_info.dwo, "e", @progbits
## CU0 uses the 32-bit DWARF format.
.LCU0:
    .long .LCU0End-.LCU0Ver         # Length
.LCU0Ver:
    .short 4                        # Version
    .long 0                         # Abbrev. offset
    .byte 8                         # Address size
    .uleb128 1                      # Abbrev [1] DW_TAG_compile_unit
    .uleb128 0                      # DW_AT_producer ("CU0 Producer")
    .uleb128 1                      # DW_AT_name ("CU0 Name")
    .quad 0x1100001122222222        # DW_AT_GNU_dwo_id
.LCU0End:
## CU1 uses the 64-bit DWARF format.
.LCU1:
    .long 0xffffffff                # DWARF64 mark
    .quad .LCU1End-.LCU1Ver           # Length
.LCU1Ver:
    .short 4                        # Version
    .quad 0                         # Abbrev. offset
    .byte 8                         # Address size
    .uleb128 1                      # Abbrev [1] DW_TAG_compile_unit
    .uleb128 0                      # DW_AT_producer ("CU1 Producer")
    .uleb128 1                      # DW_AT_name ("CU1 Name")
    .quad 0x1100001133333333        # DW_AT_GNU_dwo_id
.LCU1End:

    .section .debug_cu_index, "", @progbits
## Header:
    .long 2                         # Version
    .long 3                         # Section count
    .long 2                         # Unit count
    .long 4                         # Slot count
## Hash Table of Signatures:
    .quad 0
    .quad 0
    .quad 0x1100001122222222        # DWO Id of CU0
    .quad 0x1100001133333333        # DWO Id of CU1
## Parallel Table of Indexes:
    .long 0
    .long 0
    .long 1
    .long 2
## Table of Section Offsets:
## Row 0:
    .long 1                         # DW_SECT_INFO
    .long 3                         # DW_SECT_ABBREV
    .long 6                         # DW_SECT_STR_OFFSETS
## Row 1, offsets of contributions of CU0:
    .long .LCU0-.debug_info.dwo
    .long .LAbbr-.debug_abbrev.dwo
    .long .LSO0-.debug_str_offsets.dwo
## Row 2, offsets of contributions of CU1:
    .long .LCU1-.debug_info.dwo
    .long .LAbbr-.debug_abbrev.dwo
    .long .LSO1-.debug_str_offsets.dwo
## Table of Section Sizes:
## Row 1, sizes of contributions of CU0:
    .long .LCU0End-.LCU0
    .long .LAbbrEnd-.LAbbr
    .long .LSO0End-.LSO0
## Row 2, sizes of contributions of CU1:
    .long .LCU1End-.LCU1
    .long .LAbbrEnd-.LAbbr
    .long .LSO1End-.LSO1
