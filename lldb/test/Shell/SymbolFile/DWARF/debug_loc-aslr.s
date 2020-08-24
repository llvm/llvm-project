# This test checks the handling of location lists in the case when the module is
# not loaded at the address at which it was linked (as happens with ASLR for
# instance).

# REQUIRES: x86

# RUN: yaml2obj %S/Inputs/debug_loc-aslr.yaml -o %t.dmp
# RUN: llvm-mc --triple=x86_64-pc-linux --filetype=obj %s >%t.o
# RUN: %lldb -c %t.dmp -o "image add %t.o" \
# RUN:   -o "image load --file %t.o --slide 0x470000" \
# RUN:   -o "thread info" -o "frame variable" -o exit | FileCheck %s

# CHECK: thread #1: tid = 16001, 0x0000000000470001 {{.*}}`_start
# CHECK: (int) x = 47
# CHECK: (int) y = 74

        .text
        .globl _start
_start:
        nop
        retq
.Lstart_end:

        .section        .debug_loc,"",@progbits
# This location list implicitly uses the base address of the compile unit.
.Ldebug_loc0:
        .quad   _start-_start
        .quad   .Lstart_end-_start
        .short  3                       # Loc expr size
        .byte   8                       # DW_OP_const1u
        .byte   47
        .byte   159                     # DW_OP_stack_value
        .quad   0
        .quad   0

# This is an equivalent location list to the first one, but here the base
# address is set explicitly.
.Ldebug_loc1:
        .quad   -1
        .quad   _start
        .quad   _start-_start
        .quad   .Lstart_end-_start
        .short  3                       # Loc expr size
        .byte   8                       # DW_OP_const1u
        .byte   74
        .byte   159                     # DW_OP_stack_value
        .quad   0
        .quad   0

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
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   4                       # Abbreviation Code
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
        .byte   6                       # Abbreviation Code
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
        .byte   1                       # Abbrev [1] 0xb:0x6a DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .short  12                      # DW_AT_language
        .quad   _start                  # DW_AT_low_pc
        .long   .Lstart_end-_start      # DW_AT_high_pc
        .byte   2                       # Abbrev [2] 0x2a:0x43 DW_TAG_subprogram
        .quad   _start                  # DW_AT_low_pc
        .long   .Lstart_end-_start      # DW_AT_high_pc
        .asciz  "_start"                # DW_AT_name
        .byte   4                       # Abbrev [4] 0x52:0xf DW_TAG_variable
        .long   .Ldebug_loc0            # DW_AT_location
        .asciz  "x"                     # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   4                       # Abbrev [4] 0x52:0xf DW_TAG_variable
        .long   .Ldebug_loc1            # DW_AT_location
        .asciz  "y"                     # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   0                       # End Of Children Mark
.Lint:
        .byte   6                       # Abbrev [6] 0x6d:0x7 DW_TAG_base_type
        .asciz  "int"                   # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:
