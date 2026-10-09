# A variable whose DW_AT_type does not resolve to any DIE has no
# CompilerType. Check that a binary operator on it reports a clean
# diagnostic instead of crashing by dereferencing a type system the
# operand doesn't have.
#
# REQUIRES: aarch64-registered-target
#
# RUN: llvm-mc -triple=aarch64 -filetype=obj %s -o %t.o
# RUN: yaml2obj %S/Inputs/variable-no-type-system.yaml -o %t.dmp
# RUN: echo 'image add %t.o' > %t.cmds
# RUN: echo 'image load --file %t.o --slide 0x10000' >> %t.cmds
# RUN: echo 'frame variable global_var' >> %t.cmds
# RUN: echo 'frame variable "global_var+1"' >> %t.cmds
# RUN: %lldb -c %t.dmp -o "command source -e false %t.cmds" -o exit 2>&1 \
# RUN:   | FileCheck %s

# CHECK: frame variable global_var
# CHECK-NEXT: error:

# CHECK:      frame variable "global_var+1"
# CHECK-NEXT: {{^ +(\^|˄)}}
# CHECK-NEXT: {{^ +(╰─ )?}}error: invalid operands to binary expression ('<invalid>' and 'int')

        .text
        .globl _start
_start:
        nop
        ret
.Lstart_end:

        .section        .debug_abbrev,"",@progbits
        .byte   1                       // Abbreviation Code
        .byte   17                      // DW_TAG_compile_unit
        .byte   1                       // DW_CHILDREN_yes
        .byte   17                      // DW_AT_low_pc
        .byte   1                       // DW_FORM_addr
        .byte   18                      // DW_AT_high_pc
        .byte   6                       // DW_FORM_data4
        .byte   0                       // EOM(1)
        .byte   0                       // EOM(2)
        .byte   2                       // Abbreviation Code
        .byte   46                      // DW_TAG_subprogram
        .byte   1                       // DW_CHILDREN_yes
        .byte   17                      // DW_AT_low_pc
        .byte   1                       // DW_FORM_addr
        .byte   18                      // DW_AT_high_pc
        .byte   6                       // DW_FORM_data4
        .byte   3                       // DW_AT_name
        .byte   8                       // DW_FORM_string
        .byte   0                       // EOM(1)
        .byte   0                       // EOM(2)
        .byte   4                       // Abbreviation Code
        .byte   52                      // DW_TAG_variable
        .byte   0                       // DW_CHILDREN_no
        .byte   3                       // DW_AT_name
        .byte   8                       // DW_FORM_string
        .byte   73                      // DW_AT_type
        .byte   19                      // DW_FORM_ref4
        .byte   0                       // EOM(1)
        .byte   0                       // EOM(2)
        .byte   0                       // EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 // Length of Unit
.Ldebug_info_start0:
        .short  4                       // DWARF version number
        .long   .debug_abbrev           // Offset Into Abbrev. Section
        .byte   8                       // Address Size (in bytes)
        .byte   1                       // Abbrev [1] DW_TAG_compile_unit
        .quad   _start                  // DW_AT_low_pc
        .long   .Lstart_end-_start      // DW_AT_high_pc
        .byte   2                       // Abbrev [2] DW_TAG_subprogram
        .quad   _start                  // DW_AT_low_pc
        .long   .Lstart_end-_start      // DW_AT_high_pc
        .asciz  "frame_func"            // DW_AT_name
        .byte   4                       // Abbrev [4] DW_TAG_variable
        .asciz  "global_var"            // DW_AT_name
        // DW_AT_type: a CU-relative offset with no DIE there, as a
        // corrupted or otherwise unresolvable type reference would produce.
        .long   0x7fffffff
        .byte   0                       // End Of Children Mark
.Ldebug_info_end0:
