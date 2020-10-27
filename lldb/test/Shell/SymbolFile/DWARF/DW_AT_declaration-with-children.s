# Test that a forward-declared (DW_AT_declaration) structure is treated as a
# forward-declaration even if it has children. These types can be produced due
# to vtable-based type homing, or other -flimit-debug-info optimizations.

# REQUIRES: x86

# RUN: llvm-mc --triple x86_64-pc-linux %s --filetype=obj > %t
# RUN: %lldb %t -o "expr a" -o exit 2>&1 | FileCheck %s --check-prefix=EXPR
# RUN: %lldb %t -o "target var a" -o exit 2>&1 | FileCheck %s --check-prefix=VAR

# EXPR: incomplete type 'A' where a complete type is required

# FIXME: This should also produce some kind of an error.
# VAR: (A) a = {}

        .text
_ZN1AC2Ev:
        retq
.LZN1AC2Ev_end:

        .data
a:
        .quad   $_ZTV1A+16
        .quad   $0xdeadbeef

        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   8                               # DW_FORM_string
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
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
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   4                               # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   5                               # Abbreviation Code
        .byte   5                               # DW_TAG_formal_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   52                              # DW_AT_artificial
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   8                               # Abbreviation Code
        .byte   15                              # DW_TAG_pointer_type
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   10                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   64                              # DW_AT_frame_base
        .byte   24                              # DW_FORM_exprloc
        .byte   100                             # DW_AT_object_pointer
        .byte   19                              # DW_FORM_ref4
        .byte   71                              # DW_AT_specification
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   11                              # Abbreviation Code
        .byte   5                               # DW_TAG_formal_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   2                               # DW_AT_location
        .byte   24                              # DW_FORM_exprloc
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   52                              # DW_AT_artificial
        .byte   25                              # DW_FORM_flag_present
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
        .quad   _ZN1AC2Ev                       # DW_AT_low_pc
        .long   .LZN1AC2Ev_end-_ZN1AC2Ev        # DW_AT_high_pc
        .byte   2                               # Abbrev [2] DW_TAG_variable
        .asciz  "a"                             # DW_AT_name
        .long   .LA-.Lcu_begin0                 # DW_AT_type
        .byte   9                               # DW_AT_location
        .byte   3
        .quad   a
.LA:
        .byte   3                               # Abbrev [3] DW_TAG_structure_type
        .asciz  "A"                             # DW_AT_name
                                                # DW_AT_declaration
        .byte   4                               # Abbrev [4] DW_TAG_subprogram
        .asciz  "A"                             # DW_AT_name
                                                # DW_AT_declaration
        .byte   5                               # Abbrev [5] DW_TAG_formal_parameter
        .long   .LAptr-.Lcu_begin0              # DW_AT_type
                                                # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
.LAptr:
        .byte   8                               # Abbrev [8] DW_TAG_pointer_type
        .long   .LA-.Lcu_begin0                 # DW_AT_type
        .byte   10                              # Abbrev [10] DW_TAG_subprogram
        .quad   _ZN1AC2Ev                       # DW_AT_low_pc
        .long   .LZN1AC2Ev_end-_ZN1AC2Ev        # DW_AT_high_pc
        .byte   1                               # DW_AT_frame_base
        .byte   86
        .long   147                             # DW_AT_object_pointer
        .long   68                              # DW_AT_specification
        .byte   11                              # Abbrev [11] DW_TAG_formal_parameter
        .byte   2                               # DW_AT_location
        .byte   145
        .byte   120
        .asciz  "this"                          # DW_AT_name
        .long   .LAptr-.Lcu_begin0              # DW_AT_type
                                                # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:
