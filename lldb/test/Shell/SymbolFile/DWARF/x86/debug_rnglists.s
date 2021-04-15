# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: %lldb %t -o "image lookup -v -s lookup_rnglists" \
# RUN:   -o "image lookup -v -s lookup_rnglists2" -o exit | FileCheck %s

# CHECK-LABEL: image lookup -v -s lookup_rnglists
# CHECK:  Function: id = {0x00000030}, name = "rnglists", range = [0x0000000000000000-0x0000000000000004)
# CHECK:    Blocks: id = {0x00000030}, range = [0x00000000-0x00000004)
# CHECK-NEXT:       id = {0x00000046}, ranges = [0x00000001-0x00000002)[0x00000003-0x00000004)

# CHECK-LABEL: image lookup -v -s lookup_rnglists2
# CHECK:  Function: id = {0x0000007a}, name = "rnglists2", range = [0x0000000000000004-0x0000000000000007)
# CHECK:    Blocks: id = {0x0000007a}, range = [0x00000004-0x00000007)
# CHECK-NEXT:       id = {0x00000091}, range = [0x00000005-0x00000007)

        .text
        .p2align 12
rnglists:
        nop
.Lblock1_begin:
lookup_rnglists:
        nop
.Lblock1_end:
        nop
.Lblock2_begin:
        nop
.Lblock2_end:
.Lrnglists_end:

rnglists2:
        nop
.Lblock3_begin:
lookup_rnglists2:
        nop
        nop
.Lblock3_end:
.Lrnglists2_end:

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   116                     # DW_AT_rnglists_base
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
        .quad   rnglists                # DW_AT_low_pc
        .long   .Lrnglists_end-rnglists # DW_AT_high_pc
        .long   .Lrnglists_table_base0  # DW_AT_rnglists_base
        .byte   2                       # Abbrev [2] 0x2b:0x37 DW_TAG_subprogram
        .quad   rnglists                # DW_AT_low_pc
        .long   .Lrnglists_end-rnglists # DW_AT_high_pc
        .asciz  "rnglists"              # DW_AT_name
        .byte   5                       # Abbrev [5] 0x52:0xf DW_TAG_lexical_block
        .byte   0                       # DW_AT_ranges
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:

.Lcu_begin1:
        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   1                       # Abbrev [1] 0xc:0x5f DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .quad   rnglists2               # DW_AT_low_pc
        .long   .Lrnglists2_end-rnglists2 # DW_AT_high_pc
        .long   .Lrnglists_table_base1  # DW_AT_rnglists_base
        .byte   2                       # Abbrev [2] 0x2b:0x37 DW_TAG_subprogram
        .quad   rnglists2               # DW_AT_low_pc
        .long   .Lrnglists2_end-rnglists2 # DW_AT_high_pc
        .asciz  "rnglists2"             # DW_AT_name
        .byte   5                       # Abbrev [5] 0x52:0xf DW_TAG_lexical_block
        .byte   0                       # DW_AT_ranges
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Ldebug_info_end1:

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
        .byte   4                       # DW_RLE_offset_pair
        .uleb128 .Lblock2_begin-rnglists  #   starting offset
        .uleb128 .Lblock2_end-rnglists    #   ending offset
        .byte   0                       # DW_RLE_end_of_list
.Ldebug_rnglist_table_end0:

        .long   .Ldebug_rnglist_table_end1-.Ldebug_rnglist_table_start1 # Length
.Ldebug_rnglist_table_start1:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   1                       # Offset entry count
.Lrnglists_table_base1:
        .long   .Ldebug_ranges1-.Lrnglists_table_base1
.Ldebug_ranges1:
        .byte   4                       # DW_RLE_offset_pair
        .uleb128 .Lblock3_begin-rnglists2 #   starting offset
        .uleb128 .Lblock3_end-rnglists2   #   ending offset
        .byte   0                       # DW_RLE_end_of_list
.Ldebug_rnglist_table_end1:
