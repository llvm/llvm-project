# RUN: llvm-mc -filetype=obj -o %t -triple x86_64-pc-linux %s
# RUN: %lldb %t -o "target variable var" -b | FileCheck %s

# CHECK: (lldb) target variable var
# CHECK: (long) var = <Unhandled opcode DW_OP_unknown_ff in DWARFExpression>

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   2                       # DW_AT_location
        .byte   24                      # DW_FORM_exprloc
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
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
        .long   .Lcu_end-.Lcu_start     # Length of Unit
.Lcu_start:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x6c DW_TAG_compile_unit
.Llong:
        .byte   3                       # Abbrev [3] 0x33:0x7 DW_TAG_base_type
        .asciz  "long"                  # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   8                       # DW_AT_byte_size
        .byte   2                       # Abbrev [2] 0x3a:0x15 DW_TAG_variable
        .asciz  "var"                   # DW_AT_name
        .long   .Llong                  # DW_AT_type
        .byte   2f-1f                   # DW_AT_location
1:
        .byte   0xff                    # Invalid opcode
        .byte   0xe                     # DW_OP_constu
        .quad   0xdeadbeefbaadf00d
        .byte   0x9f                    # DW_OP_stack_value
2:
        .byte   0                       # End Of Children Mark
.Lcu_end:
