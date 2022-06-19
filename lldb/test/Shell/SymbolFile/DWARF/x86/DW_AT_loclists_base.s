# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: %lldb %t -o "image lookup -v -s lookup_loclists" -o exit | FileCheck %s

# CHECK-LABEL: image lookup -v -s lookup_loclists
# CHECK: Variable: {{.*}}, name = "x0", type = "int", valid ranges = <block>, location = [0x0000000000000000, 0x0000000000000003) -> DW_OP_reg0 RAX,
# CHECK-NOT: Variable:

loclists:
        nop
.Ltmp0:
        nop
lookup_loclists:
.Ltmp1:
        nop
.Ltmp2:
        nop
.Ltmp3:
        nop
.Ltmp4:
        nop
.Lloclists_end:

        .section        .debug_loclists,"",@progbits
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
        .byte   4                       # DW_LLE_offset_pair
        .uleb128 .Ltmp3-loclists
        .uleb128  .Ltmp4-loclists
        .uleb128 1                      # Expression size
        .byte   81                      # super-register DW_OP_reg1
        .byte   0                       # DW_LLE_end_of_list
.Ldebug_loclist_table_end0:

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .uleb128 0x8c                   # DW_AT_loclists_base
        .byte   0x17                    # DW_FORM_sec_offset
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
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

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   1                       # Abbrev [1] 0xb:0x50 DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .short  12                      # DW_AT_language
        .long   .Lloclists_table_base   # DW_AT_loclists_base
        .quad   loclists                # DW_AT_low_pc
        .long   .Lloclists_end-loclists # DW_AT_high_pc
        .byte   2                       # Abbrev [2] 0x2a:0x29 DW_TAG_subprogram
        .quad   loclists                # DW_AT_low_pc
        .long   .Lloclists_end-loclists # DW_AT_high_pc
        .asciz  "loclists"              # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .uleb128 0                      # DW_AT_location
        .asciz  "x0"                    # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .uleb128 1                      # DW_AT_location
        .asciz  "x1"                    # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
        .byte   0                       # End Of Children Mark
.Lint:
        .byte   4                       # Abbrev [4] 0x53:0x7 DW_TAG_base_type
        .asciz  "int"                   # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:
