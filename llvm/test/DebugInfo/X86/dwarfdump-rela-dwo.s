# Test objective to verify warning is printed if DWO secton has relocations.
#
# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o
# RUN: llvm-dwarfdump --debug-info %t.o | FileCheck %s
# RUN: llvm-dwarfdump --debug-info %t.o 2> %t.txt
# RUN: FileCheck --input-file=%t.txt %s --check-prefix=PART2 --implicit-check-not=warning:

        .section .debug_str.dwo,"MSe",@progbits,1
.dwo_producer:
        .asciz "Handmade DWO producer"
.dwo_CU_5:
        .asciz "V5_dwo_compile_unit"

        .section	.debug_str_offsets.dwo,"e",@progbits
        .long	Lstr_offsets_end-Lstr_offsets_start                              # Length of String Offsets Set
        Lstr_offsets_start:
	.short	5
	.short	0
	.long	.dwo_producer-.debug_str.dwo
	.long	.dwo_CU_5-.debug_str.dwo
        Lstr_offsets_end:

# And a .dwo copy for the .dwo sections.
        .section .debug_abbrev.dwo,"e",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x25  # DW_AT_producer
        .byte 0x0e  # DW_FORM_strp
        .byte 0x03  # DW_AT_name
        .byte 0x25  # DW_FORM_strx1
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)

# PART2: warning: unexpected relocations for dwo section '.debug_abbrev.dwo'
        .reloc ., R_X86_64_NONE, 0

        .section .debug_info.dwo,"e",@progbits
# CHECK-LABEL: .debug_info.dwo

# DWARF v5 split CU header.
        .long  CU_split_5_end-CU_split_5_version # Length of Unit
CU_split_5_version:
        .short 5                # DWARF version number
        .byte 5                 # DWARF Unit Type
        .byte 8                 # Address Size (in bytes)
        .long 0 # Offset Into Abbrev. Section
        .quad 0x5a              # DWO ID
# The split compile-unit DIE, with DW_AT_producer, DW_AT_name, DW_AT_stmt_list.
        .byte 1
        .long .dwo_producer
        .byte 1
        .byte 0 # NULL
CU_split_5_end:

# CHECK: 0x00000000: Compile Unit: length = 0x00000017, format = DWARF32, version = 0x0005, unit_type = DW_UT_split_compile, abbr_offset = 0x0000, addr_size = 0x08, DWO_id = 0x000000000000005a (next unit at 0x0000001b)
# CHECK: 0x00000014: DW_TAG_compile_unit
# CHECK-NEXT: DW_AT_producer	("Handmade DWO producer")
# CHECK-NEXT: DW_AT_name	("V5_dwo_compile_unit")
# PART2: warning: unexpected relocations for dwo section '.debug_info.dwo'

## No warning, even if their names end with ".dwo".
        .section .rodata.dwo,"a"
        .long ext
        .section .foo.dwo,""
        .long .dwo_producer
