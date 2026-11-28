# Test that an inlined function shows the variables that only its abstract
# origin describes, once each.

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj -o %t.o
# RUN: %lldb %t.o -o "image lookup -v -s f_in_main" -o exit | FileCheck %s

# The concrete instance inherits the location.
# CHECK:      name = "in_both", type = "int"
# CHECK-NOT:  name = "in_both"
# CHECK:      name = "located", type = "int"
# CHECK-NEXT: name = "constant", type = "int"
# CHECK-NOT:  name = "{{in_both|local}}"

        .text
        .globl  main
        .type   main,@function
main:
        nop
f_in_main:
        nop
.Lf_end:
        nop
.Lmain_end:
        .size   main, .Lmain_end-main

        .data
storage:
        .long   0
        .long   0

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
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   32                      # DW_AT_inline
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
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
        .byte   4                       # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   28                      # DW_AT_const_value
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   6                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   1                       # DW_CHILDREN_yes
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   7                       # Abbreviation Code
        .byte   29                      # DW_TAG_inlined_subroutine
        .byte   1                       # DW_CHILDREN_yes
        .byte   49                      # DW_AT_abstract_origin
        .byte   19                      # DW_FORM_ref4
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   8                       # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   49                      # DW_AT_abstract_origin
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   9                       # Abbreviation Code
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
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .short  12                      # DW_AT_language
        .quad   main                    # DW_AT_low_pc
        .long   .Lmain_end-main         # DW_AT_high_pc
.Lf:
        .byte   2                       # Abbrev [2] DW_TAG_subprogram
        .asciz  "f"                     # DW_AT_name
        .byte   1                       # DW_AT_inline
.Lin_both:
        .byte   3                       # Abbrev [3] DW_TAG_variable
        .asciz  "in_both"               # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
        .byte   9                       # DW_AT_location
        .byte   3                       # DW_OP_addr
        .quad   storage
        .byte   3                       # Abbrev [3] DW_TAG_variable
        .asciz  "located"               # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
        .byte   9                       # DW_AT_location
        .byte   3                       # DW_OP_addr
        .quad   storage+4
        .byte   4                       # Abbrev [4] DW_TAG_variable
        .asciz  "constant"              # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
        .byte   5                       # DW_AT_const_value
        .byte   5                       # Abbrev [5] DW_TAG_variable
        .asciz  "local"                 # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
        .byte   0                       # End Of Children Mark
        .byte   6                       # Abbrev [6] DW_TAG_subprogram
        .asciz  "main"                  # DW_AT_name
        .quad   main                    # DW_AT_low_pc
        .long   .Lmain_end-main         # DW_AT_high_pc
        .byte   7                       # Abbrev [7] DW_TAG_inlined_subroutine
        .long   .Lf-.Lcu_begin0         # DW_AT_abstract_origin
        .quad   f_in_main               # DW_AT_low_pc
        .long   .Lf_end-f_in_main       # DW_AT_high_pc
        .byte   8                       # Abbrev [8] DW_TAG_variable
        .long   .Lin_both-.Lcu_begin0   # DW_AT_abstract_origin
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Lint:
        .byte   9                       # Abbrev [9] DW_TAG_base_type
        .asciz  "int"                   # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:
