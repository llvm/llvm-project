# Test to verify that, if a class type pointer creation fails (pointer is
# null), LLDB does not try to dereference the null pointer.

# RUN: llvm-mc --triple x86_64-pc-linux %s --filetype=obj -o %t
# RUN: %lldb %t -o "target variable x" -o exit 2>&1 | FileCheck %s

# CHECK: 'Unable to determine byte size.'

# This tests a fix for a crash. If things are working we don't get a segfault.

        .type   x,@object                       # @x
        .bss
        .globl  x
x:
        .quad   0                               # 0x0
        .size   x, 8

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
        .byte   31                              # DW_TAG_ptr_to_member_type
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   29                              # DW_AT_containing_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   0                               # EOM(3)
        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                               # DWARF version number
        .byte   1                               # DWARF Unit Type
        .byte   8                               # Address Size (in bytes)
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   1                               # Abbrev [1] DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"            # DW_AT_producer
        .byte   2                               # Abbrev [2] DW_TAG_variable
        .asciz  "x"                             # DW_AT_name
        .long   .Ltype-.Lcu_begin0              # DW_AT_type
        .byte   9                               # DW_AT_location
        .byte   3
        .quad   x
.Ltype:
        .byte   3                               # Abbrev [3] DW_TAG_ptr_to_member_type
        .long   0xdeadbeef                      # DW_AT_type
        .long   0xdeadbeef                      # DW_AT_containing_type
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:
