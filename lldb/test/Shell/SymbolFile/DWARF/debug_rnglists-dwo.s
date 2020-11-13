# REQUIRES: x86

# RUN: cd %T
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s >debug_rnglists-dwo.o
# RUN: %lldb debug_rnglists-dwo.o -o "image lookup -v -s lookup_rnglists" \
# RUN:   -o exit | FileCheck %s

# CHECK-LABEL: image lookup -v -s lookup_rnglists
# CHECK:  Function: id = {0x4000000000000028}, name = "rnglists", range = [0x0000000000000000-0x0000000000000003)
# CHECK:    Blocks: id = {0x4000000000000028}, range = [0x00000000-0x00000003)
# CHECK-NEXT:       id = {0x4000000000000037}, range = [0x00000001-0x00000002)

        .text
rnglists:
        nop
.Lblock1_begin:
lookup_rnglists:
        nop
.Lblock1_end:
        nop
.Lrnglists_end:

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   0                       # DW_CHILDREN_no
        .byte   0x76                    # DW_AT_dwo_name
        .byte   8                       # DW_FORM_string
        .byte   115                     # DW_AT_addr_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   85                      # DW_AT_ranges
        .byte   35                      # DW_FORM_rnglistx
        .byte   116                     # DW_AT_rnglists_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                       # DWARF version number
        .byte   4                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .quad   0xdeadbeefbaadf00d      # DWO id
        .byte   1                       # Abbrev [1] 0xc:0x5f DW_TAG_compile_unit
        .asciz  "debug_rnglists-dwo.o"  # DW_AT_dwo_name
        .long   .Laddr_table_base0      # DW_AT_addr_base
        .byte   0                       # DW_AT_ranges
        .long   .Lskel_rnglists_table_base # DW_AT_rnglists_base
.Ldebug_info_end0:

        .section        .debug_addr,"",@progbits
        .long   .Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
        .short  5                       # DWARF version number
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
.Laddr_table_base0:
        .quad   rnglists
        .quad   .Lblock1_begin
.Ldebug_addr_end0:

        .section        .debug_rnglists,"",@progbits
        # A fake rnglists contribution so that range list bases for the skeleton
        # and split units differ.
        .long   .Lfake_rnglist_table_end-.Lfake_rnglist_table_start # Length
.Lfake_rnglist_table_start:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   0                       # Offset entry count
.Lfake_rnglists_table_base:
.Lfake_rnglist_table_end:

        .long   .Lskel_rnglist_table_end-.Lskel_rnglist_table_start # Length
.Lskel_rnglist_table_start:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   1                       # Offset entry count
.Lskel_rnglists_table_base:
        .long   .Lskel_ranges0-.Lskel_rnglists_table_base
.Lskel_ranges0:
        .byte   7                       # DW_RLE_start_length
        .quad   rnglists
        .uleb128   .Lrnglists_end-rnglists
        .byte   0                       # DW_RLE_end_of_list
.Lskel_rnglist_table_end:

        .section        .debug_abbrev.dwo,"e",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   1                       # DW_CHILDREN_yes
        .byte   17                      # DW_AT_low_pc
        .byte   27                      # DW_FORM_addrx
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   11                      # DW_TAG_lexical_block
        .byte   0                       # DW_CHILDREN_no
        .byte   85                      # DW_AT_ranges
        .byte   35                      # DW_FORM_rnglistx
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info.dwo,"e",@progbits
.Lcu_begin1:
        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  5                       # DWARF version number
        .byte   5                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .quad   0xdeadbeefbaadf00d      # DWO id
        .byte   1                       # Abbrev [1] 0xc:0x5f DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .byte   2                       # Abbrev [2] 0x2b:0x37 DW_TAG_subprogram
        .byte   0                       # DW_AT_low_pc
        .long   .Lrnglists_end-rnglists # DW_AT_high_pc
        .asciz  "rnglists"              # DW_AT_name
        .byte   5                       # Abbrev [5] 0x52:0xf DW_TAG_lexical_block
        .byte   0                       # DW_AT_ranges
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Ldebug_info_end1:

        .section        .debug_rnglists.dwo,"e",@progbits
        .long   .Ldwo_rnglist_table_end-.Ldwo_rnglist_table_start # Length
.Ldwo_rnglist_table_start:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   1                       # Offset entry count
.Ldwo_rnglists_table_base:
        .long   .Ldwo_ranges-.Ldwo_rnglists_table_base
.Ldwo_ranges:
        .byte   3                       # DW_RLE_startx_length
        .uleb128 1
        .uleb128 .Lblock1_end-.Lblock1_begin
        .byte   0                       # DW_RLE_end_of_list
.Ldwo_rnglist_table_end:
