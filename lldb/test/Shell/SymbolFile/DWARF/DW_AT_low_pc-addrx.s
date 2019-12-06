# REQUIRES: x86

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: %lldb %t -o "image lookup -v -s lookup_rnglists" \
# RUN:   -o exit | FileCheck %s

# CHECK-LABEL: image lookup -v -s lookup_rnglists
# CHECK:  Function: id = {0x7fffffff0000002d}, name = "rnglists", range = [0x0000000000000000-0x0000000000000003)
# CHECK:    Blocks: id = {0x7fffffff0000002d}, range = [0x00000000-0x00000003)
# CHECK-NEXT:       id = {0x7fffffff00000043}, range = [0x00000001-0x00000002)

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
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   17                      # DW_AT_low_pc
        .byte   27                      # DW_FORM_addrx
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   116                     # DW_AT_rnglists_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   115                     # DW_AT_addr_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   1                       # DW_CHILDREN_yes
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
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

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   1                       # Abbrev [1] 0xc:0x5f DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .byte   0                       # DW_AT_low_pc
        .long   .Lrnglists_end-rnglists # DW_AT_high_pc
        .long   .Lrnglists_table_base0  # DW_AT_rnglists_base
        .long   .Laddr_table_base0      # DW_AT_addr_base
        .byte   2                       # Abbrev [2] 0x2b:0x37 DW_TAG_subprogram
        .quad   rnglists                # DW_AT_low_pc
        .long   .Lrnglists_end-rnglists # DW_AT_high_pc
        .asciz  "rnglists"              # DW_AT_name
        .byte   5                       # Abbrev [5] 0x52:0xf DW_TAG_lexical_block
        .byte   0                       # DW_AT_ranges
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:

        .section        .debug_addr,"",@progbits
        .long   .Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
        .short  5                       # DWARF version number
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
.Laddr_table_base0:
        .quad   rnglists
.Ldebug_addr_end0:

        .section        .debug_rnglists,"",@progbits
        .long   .Ldebug_rnglist_table_end0-.Ldebug_rnglist_table_start0 # Length
.Ldebug_rnglist_table_start0:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   1                       # Offset entry count
.Lrnglists_table_base0:
        .long   .Ldebug_ranges0-.Lrnglists_table_base0
.Ldebug_ranges0:
        .byte   4                       # DW_RLE_offset_pair
        .uleb128 .Lblock1_begin-rnglists  #   starting offset
        .uleb128 .Lblock1_end-rnglists    #   ending offset
        .byte   0                       # DW_RLE_end_of_list
.Ldebug_rnglist_table_end0:
