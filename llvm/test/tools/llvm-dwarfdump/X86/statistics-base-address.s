# This tests the computation of the scope bytes covered by local variables. In
# particular, the case when the variable starts in the middle of the enclosing
# scope, and the compile unit has both DW_AT_ranges and DW_AT_low_pc attributes.

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj -o %t
# RUN: llvm-dwarfdump --statistics %t | FileCheck %s

# CHECK: "vars scope bytes total":12
# CHECK: "vars scope bytes covered":8

        .text

# Add padding to ensure the function does not start at address zero.
        .zero 256

f:                                      # @f
.Lf_begin:
        .zero 4
.Lx_begin:
        .zero 8
.Lf_end:

        .section        .debug_ranges,"",@progbits
.Ldebug_ranges:
        .quad   .Lf_begin
        .quad   .Lf_end
        .quad   0
        .quad   0

        .section        .debug_loc,"",@progbits
.Ldebug_loc:
        .quad   .Lx_begin
        .quad   .Lf_end
        .short  1                       # Loc expr size
        .byte   85                      # super-register DW_OP_reg5
        .quad   0
        .quad   0


        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   85                      # DW_AT_ranges
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
        .byte   3                       # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   2                       # DW_AT_location
        .byte   23                      # DW_FORM_sec_offset
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
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
.Lcu_begin:
        .long   .Ldebug_info_end-.Ldebug_info_start # Length of Unit
.Ldebug_info_start:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x64 DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .quad   0                       # DW_AT_low_pc
        .long   .Ldebug_ranges          # DW_AT_ranges
        .byte   2                       # Abbrev [2] 0x2a:0x28 DW_TAG_subprogram
        .quad   .Lf_begin               # DW_AT_low_pc
        .long   .Lf_end-.Lf_begin       # DW_AT_high_pc
        .asciz  "f"                     # DW_AT_name
        .byte   3                       # Abbrev [3] 0x43:0xe DW_TAG_variable
        .long   .Ldebug_loc             # DW_AT_location
        .asciz  "x"                     # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   0                       # End Of Children Mark
.Lint:
        .byte   5                       # Abbrev [5] 0x67:0x7 DW_TAG_base_type
        .asciz  "int"                   # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Ldebug_info_end:
