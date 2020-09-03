# Test that we can handle DWARF 4 and 5 location lists in the same object file
# (but different compile units).

# REQUIRES: x86

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: %lldb %t -o "image lookup -v -s loc" -o "image lookup -v -s loclists" \
# RUN:   -o exit | FileCheck %s


# CHECK-LABEL: image lookup -v -s loc
# CHECK: Variable: {{.*}}, name = "x0", type = "int", location = DW_OP_reg5 RDI,

# CHECK-LABEL: image lookup -v -s loclists
# CHECK: Variable: {{.*}}, name = "x1", type = "int", location = DW_OP_reg0 RAX,


loc:
        nop
.Lloc_end:

loclists:
        nop
.Lloclists_end:

        .section        .debug_loc,"",@progbits
.Lloc_list:
        .quad loc-loc
        .quad .Lloc_end-loc
        .short 1
        .byte   85                      # super-register DW_OP_reg5
        .quad 0
        .quad 0

        .section        .debug_loclists,"",@progbits
        .long   .Ldebug_loclist_table_end0-.Ldebug_loclist_table_start0 # Length
.Ldebug_loclist_table_start0:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   0                       # Offset entry count

.Lloclists_list:
        .byte   4                       # DW_LLE_offset_pair
        .uleb128 loclists-loclists
        .uleb128 .Lloclists_end-loclists
        .uleb128 1
        .byte   80                      # super-register DW_OP_reg0
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
        .byte   16                      # DW_FORM_ref_addr
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   5                       # DW_TAG_formal_parameter
        .byte   0                       # DW_CHILDREN_no
        .byte   2                       # DW_AT_location
        .byte   23                      # DW_FORM_sec_offset
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   16                      # DW_FORM_ref_addr
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
        .long   .Lloc_cu_end-.Lloc_cu_start # Length of Unit
.Lloc_cu_start:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x50 DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .short  12                      # DW_AT_language
        .quad   loc                     # DW_AT_low_pc
        .long   .Lloc_end-loc           # DW_AT_high_pc
        .byte   2                       # Abbrev [2] 0x2a:0x29 DW_TAG_subprogram
        .quad   loc                     # DW_AT_low_pc
        .long   .Lloc_end-loc           # DW_AT_high_pc
        .asciz  "loc"                   # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   .Lloc_list              # DW_AT_location
        .asciz  "x0"                    # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   0                       # End Of Children Mark
.Lint:
        .byte   4                       # Abbrev [4] 0x53:0x7 DW_TAG_base_type
        .asciz  "int"                   # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Lloc_cu_end:

        .long   .Lloclists_cu_end-.Lloclists_cu_start # Length of Unit
.Lloclists_cu_start:
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   1                       # Abbrev [1] 0xb:0x50 DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .short  12                      # DW_AT_language
        .quad   loclists                # DW_AT_low_pc
        .long   .Lloclists_end-loclists # DW_AT_high_pc
        .byte   2                       # Abbrev [2] 0x2a:0x29 DW_TAG_subprogram
        .quad   loclists                # DW_AT_low_pc
        .long   .Lloclists_end-loclists # DW_AT_high_pc
        .asciz  "loclists"              # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   .Lloclists_list         # DW_AT_location
        .asciz  "x1"                    # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Lloclists_cu_end:
