## Test that we don't get confused by variables with both location and
## const_value attributes. Such values are produced in C++ for class-level
## static constexpr variables.

# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o %t
# RUN: %lldb %t -o "target variable A::x A::y" -o exit | FileCheck %s

# CHECK-LABEL: target variable
# CHECK: (const int) A::x = 142
# CHECK: (const int) A::y = 242

        .section        .rodata,"a",@progbits
        .p2align        2
_ZN1A1xE:
        .long   142
_ZN1A1yE:
        .long   242

        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   8                               # DW_FORM_string
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   3                               # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   4                               # Abbreviation Code
        .byte   13                              # DW_TAG_member
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   28                              # DW_AT_const_value
        .byte   13                              # DW_FORM_sdata
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   5                               # Abbreviation Code
        .byte   38                              # DW_TAG_const_type
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   6                               # Abbreviation Code
        .byte   36                              # DW_TAG_base_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   62                              # DW_AT_encoding
        .byte   11                              # DW_FORM_data1
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   7                               # Abbreviation Code
        .byte   52                              # DW_TAG_variable
        .byte   0                               # DW_CHILDREN_no
        .byte   71                              # DW_AT_specification
        .byte   19                              # DW_FORM_ref4
        .byte   2                               # DW_AT_location
        .byte   24                              # DW_FORM_exprloc
        .byte   110                             # DW_AT_linkage_name
        .byte   8                               # DW_FORM_string
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
## This deliberately inverts the order of the specification and location
## attributes.
        .byte   8                               # Abbreviation Code
        .byte   52                              # DW_TAG_variable
        .byte   0                               # DW_CHILDREN_no
        .byte   2                               # DW_AT_location
        .byte   24                              # DW_FORM_exprloc
        .byte   71                              # DW_AT_specification
        .byte   19                              # DW_FORM_ref4
        .byte   110                             # DW_AT_linkage_name
        .byte   8                               # DW_FORM_string
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   0                               # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                               # DWARF version number
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   8                               # Address Size (in bytes)
        .byte   1                               # Abbrev DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"            # DW_AT_producer
        .asciz  "a.cc"                          # DW_AT_name
        .byte   7                               # Abbrev DW_TAG_variable
        .long   .LA__x-.Lcu_begin0              # DW_AT_specification
        .byte   9                               # DW_AT_location
        .byte   3
        .quad   _ZN1A1xE
        .asciz  "_ZN1A1xE"                      # DW_AT_linkage_name
        .byte   8                               # Abbrev DW_TAG_variable
        .byte   9                               # DW_AT_location
        .byte   3
        .quad   _ZN1A1yE
        .long   .LA__y-.Lcu_begin0              # DW_AT_specification
        .asciz  "_ZN1A1yE"                      # DW_AT_linkage_name
        .byte   3                               # Abbrev DW_TAG_structure_type
        .asciz  "A"                             # DW_AT_name
        .byte   1                               # DW_AT_byte_size
.LA__x:
        .byte   4                               # Abbrev DW_TAG_member
        .asciz  "x"                             # DW_AT_name
        .long   .Lconst_int-.Lcu_begin0         # DW_AT_type
                                                # DW_AT_declaration
        .sleb128 147                            # DW_AT_const_value
.LA__y:
        .byte   4                               # Abbrev DW_TAG_member
        .asciz  "y"                             # DW_AT_name
        .long   .Lconst_int-.Lcu_begin0         # DW_AT_type
                                                # DW_AT_declaration
        .sleb128 247                            # DW_AT_const_value
        .byte   0                               # End Of Children Mark
.Lconst_int:
        .byte   5                               # Abbrev DW_TAG_const_type
        .long   .Lint-.Lcu_begin0               # DW_AT_type
.Lint:
        .byte   6                               # Abbrev DW_TAG_base_type
        .asciz  "int"                           # DW_AT_name
        .byte   5                               # DW_AT_encoding
        .byte   4                               # DW_AT_byte_size
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:
