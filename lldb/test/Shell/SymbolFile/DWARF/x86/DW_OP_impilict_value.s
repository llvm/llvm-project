# Test that LLDB correctly handles DW_OP_implicit_value for scalar
# and aggregate types.
#
# RUN: llvm-mc -filetype=obj -o %t -triple x86_64-pc-linux %s
# RUN: %lldb %t \
# RUN:   -o "target variable int_val" \
# RUN:   -o "target variable point" \
# RUN:   -o "target variable char_val" \
# RUN:   -b 2>&1 | FileCheck %s

# CHECK:      (lldb) target variable int_val
# CHECK-NOT: error:
# CHECK-NEXT: (int) int_val = 42

# CHECK:      (lldb) target variable point
# CHECK-NOT: error:
# CHECK-NEXT: (Point) point = {
# CHECK-NEXT:   x = 10
# CHECK-NEXT:   y = 20
# CHECK-NEXT: }

# CHECK:      (lldb) target variable char_val
# CHECK-NOT: error:
# CHECK-NEXT: (char) char_val = 'A'

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbrev [1] DW_TAG_compile_unit
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   19                      # DW_AT_language
        .byte   11                      # DW_FORM_data1
        .byte   0
        .byte   0

        .byte   2                       # Abbrev [2] DW_TAG_variable
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   2                       # DW_AT_location
        .byte   24                      # DW_FORM_exprloc
        .byte   0
        .byte   0

        .byte   3                       # Abbrev [3] DW_TAG_base_type
        .byte   36                      # DW_TAG_base_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   62                      # DW_AT_encoding
        .byte   11                      # DW_FORM_data1
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0
        .byte   0

        .byte   4                       # Abbrev [4] DW_TAG_structure_type
        .byte   19                      # DW_TAG_structure_type
        .byte   1                       # DW_CHILDREN_yes
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0
        .byte   0

        .byte   5                       # Abbrev [5] DW_TAG_member
        .byte   13                      # DW_TAG_member
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   56                      # DW_AT_data_member_location
        .byte   11                      # DW_FORM_data1
        .byte   0
        .byte   0

        .byte   0                       # End of abbrev table

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Lcu_end0 - .Lcu_start0
.Lcu_start0:
        .short  5                       # DWARF version 5
        .byte   1                       # DW_UT_compile
        .byte   8                       # Address size
        .long   .debug_abbrev

        .byte   1                       # DW_TAG_compile_unit
        .byte   12                      # DW_LANG_C99

# ---- Base types ----
.Lint_type:
        .byte   3                       # DW_TAG_base_type
        .asciz  "int"
        .byte   5                       # DW_ATE_signed
        .byte   4                       # 4 bytes

.Lchar_type:
        .byte   3                       # DW_TAG_base_type
        .asciz  "char"
        .byte   6                       # DW_ATE_signed_char
        .byte   1                       # 1 byte

.Llong_type:
        .byte   3                       # DW_TAG_base_type
        .asciz  "long"
        .byte   5                       # DW_ATE_signed
        .byte   8                       # 8 bytes

.Lstruct_type:
        .byte   4                       # DW_TAG_structure_type
        .asciz  "Point"
        .byte   8                       # byte_size = 8

        .byte   5                       # DW_TAG_member: x
        .asciz  "x"
        .long   .Lint_type - .Lcu_begin0
        .byte   0                       # offset 0

        .byte   5                       # DW_TAG_member: y
        .asciz  "y"
        .long   .Lint_type - .Lcu_begin0
        .byte   4                       # offset 4

        .byte   0                       # end of struct children

# int int_val = 42
        .byte   2                       # DW_TAG_variable
        .asciz  "int_val"
        .long   .Lint_type - .Lcu_begin0
        .byte   .Lint_loc_end - .Lint_loc_start
.Lint_loc_start:
        .byte   0x9e                    # DW_OP_implicit_value
        .uleb128 4                      # length = 4 bytes
        .long   42                      # value = 42
.Lint_loc_end:

# Point point = {10, 20}
        .byte   2                       # DW_TAG_variable
        .asciz  "point"
        .long   .Lstruct_type - .Lcu_begin0
        .byte   .Lpoint_loc_end - .Lpoint_loc_start
.Lpoint_loc_start:
        .byte   0x9e                    # DW_OP_implicit_value
        .uleb128 8                      # length = 8 bytes (two ints)
        .long   10                      # x = 10
        .long   20                      # y = 20
.Lpoint_loc_end:

# char char_val = 'A'
        .byte   2                       # DW_TAG_variable
        .asciz  "char_val"
        .long   .Lchar_type - .Lcu_begin0
        .byte   .Lchar_loc_end - .Lchar_loc_start
.Lchar_loc_start:
        .byte   0x9e                    # DW_OP_implicit_value
        .uleb128 1                      # length = 1 byte
        .byte   0x41                    # value = 'A'
.Lchar_loc_end:

        .byte   0                       # End of compile unit children
.Lcu_end0:
