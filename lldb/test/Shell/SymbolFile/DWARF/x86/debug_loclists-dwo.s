# RUN: cd %T
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s >debug_loclists-dwo.o
# RUN: %lldb debug_loclists-dwo.o -o "image lookup -v -s lookup_loclists" -o exit | FileCheck %s

# CHECK-LABEL: image lookup -v -s lookup_loclists
# CHECK: Variable: {{.*}}, name = "x0", type = "int", location = DW_OP_reg0 RAX,
# CHECK: Variable: {{.*}}, name = "x1", type = "int", location = DW_OP_reg1 RDX,

loclists:
        nop
.Ltmp0:
        nop
.Ltmp1:
lookup_loclists:
        nop
.Ltmp2:
        nop
.Ltmp3:
        nop
.Lloclists_end:

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   0                       # DW_CHILDREN_no
        .byte   0x76                    # DW_AT_dwo_name
        .byte   8                       # DW_FORM_string
        .byte   115                     # DW_AT_addr_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
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
        .asciz  "debug_loclists-dwo.o"  # DW_AT_dwo_name
        .long   .Laddr_table_base0      # DW_AT_addr_base
        .quad   loclists                # DW_AT_low_pc
        .byte   0                       # DW_AT_ranges
        .long   .Lskel_rnglists_table_base # DW_AT_rnglists_base
.Ldebug_info_end0:

        .section        .debug_rnglists,"",@progbits
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
        .quad   loclists
        .uleb128   .Lloclists_end-loclists
        .byte   0                       # DW_RLE_end_of_list
.Lskel_rnglist_table_end:
        .section        .debug_addr,"",@progbits
        .long   .Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
        .short  5                       # DWARF version number
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
.Laddr_table_base0:
        .quad   loclists
        .quad   .Ltmp1
.Ldebug_addr_end0:

# The presence of an extra non-dwo loclists section should not confuse us.
# .debug_info.dwo always refers to .debug_loclists.dwo
        .section        .debug_loclists,"",@progbits
        .quad 0xdeadbeefbaadf00d

        .section        .debug_loclists.dwo,"e",@progbits
        .long   .Ldebug_loclist_table_end0-.Ldebug_loclist_table_start0 # Length
.Ldebug_loclist_table_start0:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   2                       # Offset entry count
.Lloclists_table_base:
        .long   .Ldebug_loc0-.Lloclists_table_base
        .long   .Ldebug_loc1-.Lloclists_table_base
.Ldebug_loc0:
        .byte   4                       # DW_LLE_offset_pair
        .uleb128 loclists-loclists
        .uleb128  .Ltmp2-loclists
        .uleb128 1                      # Expression size
        .byte   80                      # super-register DW_OP_reg0
        .byte   0                       # DW_LLE_end_of_list
.Ldebug_loc1:
        .byte   3                       # DW_LLE_startx_length
        .uleb128 1
        .uleb128  .Ltmp3-.Ltmp1
        .uleb128 1                      # Expression size
        .byte   81                      # super-register DW_OP_reg1
        .byte   0                       # DW_LLE_end_of_list
.Ldebug_loclist_table_end0:

        .section        .debug_abbrev.dwo,"e",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
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
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   5                       # DW_TAG_formal_parameter
        .byte   0                       # DW_CHILDREN_no
        .byte   2                       # DW_AT_location
        .byte   0x22                    # DW_FORM_loclistx
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   4                       # Abbreviation Code
        .byte   36                      # DW_TAG_base_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   62                      # DW_AT_encoding
        .byte   11                      # DW_FORM_data1
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
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
        .byte   1                       # Abbrev [1] 0xb:0x50 DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .short  12                      # DW_AT_language
        .byte   2                       # Abbrev [2] 0x2a:0x29 DW_TAG_subprogram
        .byte   0                       # DW_AT_low_pc
        .long   .Lloclists_end-loclists # DW_AT_high_pc
        .asciz  "loclists"              # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .uleb128 0                      # DW_AT_location
        .asciz  "x0"                    # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .uleb128 1                      # DW_AT_location
        .asciz  "x1"                    # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   0                       # End Of Children Mark
.Lint:
        .byte   4                       # Abbrev [4] 0x53:0x7 DW_TAG_base_type
        .asciz  "int"                   # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Ldebug_info_end1:
