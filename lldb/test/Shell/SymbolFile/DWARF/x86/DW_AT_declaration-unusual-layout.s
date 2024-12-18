## Test that lldb respects the layout defined in DWARF even when starting out
## with a declaration of the class.

# RUN: split-file %s %t
# RUN: llvm-mc --triple x86_64-pc-linux %t/asm --filetype=obj -o %t.o
# RUN: %lldb -s %t/commands -o exit %t.o 2>&1 | FileCheck %s

#--- commands
target var a -fx
# CHECK-LABEL: target var a
# CHECK: (A) a = (i = 0xbaadf00d)

#--- asm
        .data
        .p2align 4
        .long 0
a:
        .long  0xdeadbeef
        .long  0xbaadf00d

        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   8                               # DW_FORM_string
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   2                               # Abbreviation Code
        .byte   52                              # DW_TAG_variable
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   2                               # DW_AT_location
        .byte   24                              # DW_FORM_exprloc
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   3                               # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   4                               # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   5                               # Abbreviation Code
        .byte   13                              # DW_TAG_member
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   56                              # DW_AT_data_member_location
        .byte   11                              # DW_FORM_data1
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
        .byte   0                               # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                               # DWARF version number
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   8                               # Address Size (in bytes)
        .byte   1                               # Abbrev [1] DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"            # DW_AT_producer

        .byte   2                               # Abbrev [2] DW_TAG_variable
        .asciz  "a"                             # DW_AT_name
        .long   .LA_decl-.Lcu_begin0            # DW_AT_type
        .byte   9                               # DW_AT_location
        .byte   3
        .quad   a
.LA_decl:
        .byte   3                               # Abbrev [3] DW_TAG_structure_type
        .asciz  "A"                             # DW_AT_name
                                                # DW_AT_declaration
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:

.Lcu_begin1:
        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  4                               # DWARF version number
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   8                               # Address Size (in bytes)
        .byte   1                               # Abbrev [1] DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"            # DW_AT_producer

        .byte   4                               # Abbrev [4] DW_TAG_structure_type
        .asciz  "A"                             # DW_AT_name
                                                # DW_AT_declaration
        .byte   8                               # DW_AT_byte_size
        .byte   5                               # Abbrev [5] DW_TAG_member
        .asciz  "i"                             # DW_AT_name
        .long   .Lint-.Lcu_begin1               # DW_AT_type
## NB: empty padding before this member
        .byte   4                               # DW_AT_data_member_location
        .byte   0                               # End Of Children Mark

.Lint:
        .byte   6                               # Abbrev [6] DW_TAG_base_type
        .asciz  "int"                           # DW_AT_name
        .byte   5                               # DW_AT_encoding
        .byte   4                               # DW_AT_byte_size

        .byte   0                               # End Of Children Mark
.Ldebug_info_end1:
