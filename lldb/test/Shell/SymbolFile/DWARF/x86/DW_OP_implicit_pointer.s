# Test DW_OP_implicit_pointer for both scalar types (int, char) and struct types.
#
# RUN: llvm-mc -filetype=obj -o %t -triple x86_64-pc-linux %s
# RUN: %lldb %t \
# RUN:   -o "target variable int_val" \
# RUN:   -o "target variable *int_ptr" \
# RUN:   -o "target variable char_val" \
# RUN:   -o "target variable *char_ptr" \
# RUN:   -o "target variable point" \
# RUN:   -o "target variable *struct_ptr" \
# RUN:   -o "target variable struct_ptr->y" \
# RUN:   -b | FileCheck %s

# CHECK:      (lldb) target variable int_val
# CHECK:      (int) int_val = 42

# CHECK:      (lldb) target variable *int_ptr
# CHECK:      (int) {{.*}} = 42

# CHECK:      (lldb) target variable char_val
# CHECK:      (char) char_val = 'A'

# CHECK:      (lldb) target variable *char_ptr
# CHECK:      (char) {{.*}} = 'A'

# CHECK:      (lldb) target variable point
# CHECK:      (Point) point = {
# CHECK-NEXT:   x = 10
# CHECK-NEXT:   y = 20
# CHECK-NEXT: }

# CHECK:      (lldb) target variable *struct_ptr
# CHECK:      (Point) {{.*}} = {
# CHECK-NEXT:   x = 10
# CHECK-NEXT:   y = 20
# CHECK-NEXT: }

# CHECK:      (lldb) target variable struct_ptr->y
# CHECK:      (int) {{.*}} = 20

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbrev [1] DW_TAG_compile_unit
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
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

        .byte   4                       # Abbrev [4] DW_TAG_pointer_type
        .byte   15                      # DW_TAG_pointer_type
        .byte   0                       # DW_CHILDREN_no
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0
        .byte   0

        .byte   5                       # Abbrev [5] DW_TAG_structure_type
        .byte   19                      # DW_TAG_structure_type
        .byte   1                       # DW_CHILDREN_yes
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0
        .byte   0

        .byte   6                       # Abbrev [6] DW_TAG_member
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

# ===================================================================
        .section        .debug_info,"",@progbits
.Ldummy_cu_begin:
        .long   .Ldummy_cu_end - .Ldummy_cu_start
.Ldummy_cu_start:
        .short  5                       # DWARF version 5
        .byte   1                       # DW_UT_compile
        .byte   8                       # Address size
        .long   .debug_abbrev

        .byte   1                       # DW_TAG_compile_unit
        .byte   0                       # End of compile unit children
.Ldummy_cu_end:

.Lcu_begin0:
        .long   .Lcu_end0 - .Lcu_start0
.Lcu_start0:
        .short  5                       # DWARF version 5
        .byte   1                       # DW_UT_compile
        .byte   8                       # Address size
        .long   .debug_abbrev

        .byte   1                       # DW_TAG_compile_unit

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

# ---- Pointer types ----
.Lint_ptr_type:
        .byte   4                       # DW_TAG_pointer_type
        .long   .Lint_type - .Lcu_begin0

.Lchar_ptr_type:
        .byte   4                       # DW_TAG_pointer_type
        .long   .Lchar_type - .Lcu_begin0

.Lstruct_ptr_type:
        .byte   4                       # DW_TAG_pointer_type
        .long   .Lstruct_type - .Lcu_begin0

# ---- struct Point { int x; int y; } ----
.Lstruct_type:
        .byte   5                       # DW_TAG_structure_type
        .asciz  "Point"
        .byte   8                       # byte_size = 8

        .byte   6                       # DW_TAG_member
        .asciz  "x"
        .long   .Lint_type - .Lcu_begin0
        .byte   0                       # offset 0

        .byte   6                       # DW_TAG_member
        .asciz  "y"
        .long   .Lint_type - .Lcu_begin0
        .byte   4                       # offset 4

        .byte   0                       # end of struct children

# int_val = 42
.Lint_val:
        .byte   2                       # DW_TAG_variable
        .asciz  "int_val"
        .long   .Lint_type - .Lcu_begin0
        .byte   .Lint_val_loc_end - .Lint_val_loc_start
.Lint_val_loc_start:
        .byte   0x9e                    # DW_OP_implicit_value
        .uleb128 4                      # length = 4
        .long   42
.Lint_val_loc_end:

# char_val = 'A' (0x41)
.Lchar_val:
        .byte   2                       # DW_TAG_variable
        .asciz  "char_val"
        .long   .Lchar_type - .Lcu_begin0
        .byte   .Lchar_val_loc_end - .Lchar_val_loc_start
.Lchar_val_loc_start:
        .byte   0x9e                    # DW_OP_implicit_value
        .uleb128 1                      # length = 1
        .byte   0x41                    # 'A'
.Lchar_val_loc_end:

# point = {x=10, y=20}
.Lpoint_val:
        .byte   2                       # DW_TAG_variable
        .asciz  "point"
        .long   .Lstruct_type - .Lcu_begin0
        .byte   .Lpoint_loc_end - .Lpoint_loc_start
.Lpoint_loc_start:
        .byte   0x9e                    # DW_OP_implicit_value
        .uleb128 8                      # length = 8 (two ints)
        .long   10                      # x = 10
        .long   20                      # y = 20
.Lpoint_loc_end:

# int *int_ptr -> points to int_val, offset 0
        .byte   2                       # DW_TAG_variable
        .asciz  "int_ptr"
        .long   .Lint_ptr_type - .Lcu_begin0
        .byte   .Lint_ptr_loc_end - .Lint_ptr_loc_start
.Lint_ptr_loc_start:
        .byte   0xa0                    # DW_OP_implicit_pointer
        .long   .Lint_val               # reference to int_val DIE
        .sleb128 0                      # byte offset = 0
.Lint_ptr_loc_end:

# char *char_ptr -> points to char_val, offset 0
        .byte   2                       # DW_TAG_variable
        .asciz  "char_ptr"
        .long   .Lchar_ptr_type - .Lcu_begin0
        .byte   .Lchar_ptr_loc_end - .Lchar_ptr_loc_start
.Lchar_ptr_loc_start:
        .byte   0xa0                    # DW_OP_implicit_pointer
        .long   .Lchar_val              # reference to char_val DIE
        .sleb128 0                      # byte offset = 0
.Lchar_ptr_loc_end:

# Point *struct_ptr -> points to point, offset 0
        .byte   2                       # DW_TAG_variable
        .asciz  "struct_ptr"
        .long   .Lstruct_ptr_type - .Lcu_begin0
        .byte   .Lstruct_ptr_loc_end - .Lstruct_ptr_loc_start
.Lstruct_ptr_loc_start:
        .byte   0xa0                    # DW_OP_implicit_pointer
        .long   .Lpoint_val             # reference to point DIE
        .sleb128 0                      # byte offset = 0
.Lstruct_ptr_loc_end:

        .byte   0                       # End of compile unit children
.Lcu_end0:
