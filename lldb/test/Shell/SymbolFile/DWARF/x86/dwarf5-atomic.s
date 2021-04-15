# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: %lldb %t -o "target variable integer structure" -o exit | FileCheck %s

# CHECK: (_Atomic(int)) integer = 14159
# CHECK: (_Atomic(struct_type)) structure = (member = 71828)

        .data
integer:
        .long 14159
structure:
        .long 71828

        .section        .debug_abbrev,"",@progbits
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
        .byte   71                      # DW_TAG_atomic_type
        .byte   0                       # DW_CHILDREN_no
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
        .byte   5                       # Abbreviation Code
        .byte   19                      # DW_TAG_structure_type
        .byte   1                       # DW_CHILDREN_yes
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   6                       # Abbreviation Code
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
        .byte   2                       # Abbrev [2] DW_TAG_variable
        .asciz  "integer"               # DW_AT_name
        .long   .Latomic_int            # DW_AT_type
        .byte   9                       # DW_AT_location
        .byte   3
        .quad   integer
        .byte   2                       # Abbrev [2] DW_TAG_variable
        .asciz  "structure"             # DW_AT_name
        .long   .Latomic_struct         # DW_AT_type
        .byte   9                       # DW_AT_location
        .byte   3
        .quad   structure
.Latomic_int:
        .byte   3                       # Abbrev [3] DW_TAG_atomic_type
        .long   .Lint                   # DW_AT_type
.Lint:
        .byte   4                       # Abbrev [4] 0x53:0x7 DW_TAG_base_type
        .asciz  "int"                   # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
.Latomic_struct:
        .byte   3                       # Abbrev [3] DW_TAG_atomic_type
        .long   .Lstruct                # DW_AT_type
.Lstruct:
        .byte   5                       # Abbrev [5] DW_TAG_structure_type
        .asciz  "struct_type"           # DW_AT_name
        .byte   4                       # DW_AT_byte_size
        .byte   6                       # Abbrev [6] DW_TAG_member
        .asciz  "member"                # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   0                       # DW_AT_data_member_location
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:
