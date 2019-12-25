# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-info - | \
# RUN:   FileCheck %s

        .section .debug_abbrev.dwo,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x13  # DW_AT_language
        .byte 0x05  # DW_FORM_data2
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)

# The CU was considered valid even though there were some required fields
# missing in the header.
        .section .debug_info.dwo,"",@progbits
        .long .CUend - .CUver   # Length of Unit
.CUver:
        .short 5                # DWARF version number
        .byte 5                 # DW_UT_split_compile
        .byte 4                 # Address Size (in bytes)
        # .long 0               # Missing: Offset Into Abbrev Section
        # .quad 0               # Missing: DWO id
        .byte 1                 # Abbreviation code
        .short 4                # DW_LANG_C_plus_plus
.CUend:

# CHECK-NOT: Compile Unit:
