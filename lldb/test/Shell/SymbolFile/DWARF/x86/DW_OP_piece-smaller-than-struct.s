# RUN: llvm-mc -filetype=obj -o %t -triple x86_64-pc-linux %s
# RUN: %lldb %t -o "target variable reset" -b | FileCheck %s

# CHECK: (lldb) target variable reset
# CHECK: (auto_reset) reset = {
# CHECK:   ptr = 0xdeadbeefbaadf00d
# Note: We need to find some way to represent "prev" as unknown/undefined.
# CHECK:   prev = false
# CHECK: }

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
        .byte   4                       # Abbreviation Code
        .byte   19                      # DW_TAG_structure_type
        .byte   1                       # DW_CHILDREN_yes
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   13                      # DW_TAG_member
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   56                      # DW_AT_data_member_location
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   6                       # Abbreviation Code
        .byte   15                      # DW_TAG_pointer_type
        .byte   0                       # DW_CHILDREN_no
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
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
.Lbool:
        .byte   3                       # Abbrev [3] 0x33:0x7 DW_TAG_base_type
        .asciz  "bool"                  # DW_AT_name
        .byte   2                       # DW_AT_encoding
        .byte   1                       # DW_AT_byte_size
        .byte   2                       # Abbrev [2] 0x3a:0x15 DW_TAG_variable
        .asciz  "reset"                 # DW_AT_name
        .long   .Lstruct                # DW_AT_type
        .byte   2f-1f                   # DW_AT_location
1:
        .byte   0xe                     # DW_OP_constu
        .quad   0xdeadbeefbaadf00d
        .byte   0x9f                    # DW_OP_stack_value
        .byte   0x93                    # DW_OP_piece
        .uleb128 8
        # Note: Only the first 8 bytes of the struct are described.
2:
.Lstruct:
        .byte   4                       # Abbrev [4] 0x4f:0x22 DW_TAG_structure_type
        .asciz  "auto_reset"            # DW_AT_name
        .byte   16                      # DW_AT_byte_size
        .byte   5                       # Abbrev [5] 0x58:0xc DW_TAG_member
        .asciz  "ptr"                   # DW_AT_name
        .long   .Lbool_ptr              # DW_AT_type
        .byte   0                       # DW_AT_data_member_location
        .byte   5                       # Abbrev [5] 0x64:0xc DW_TAG_member
        .asciz  "prev"                  # DW_AT_name
        .long   .Lbool                  # DW_AT_type
        .byte   8                       # DW_AT_data_member_location
        .byte   0                       # End Of Children Mark
.Lbool_ptr:
        .byte   6                       # Abbrev [6] 0x71:0x5 DW_TAG_pointer_type
        .long   .Lbool                  # DW_AT_type
        .byte   0                       # End Of Children Mark
.Lcu_end:
