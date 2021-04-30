# This tests handling of debug info with fully relative paths, such as those
# produced by "clang -fdebug-compilation-dir <something-relative>". This is one
# of the techniques used to produce "relocatable" debug info.

# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux -o %t %s
# RUN: %lldb %t -o "image dump line-table t.c" | FileCheck %s

# CHECK: 0x0000000000000000: {{q[\\/]w[\\/]e[\\/]r[\\/]t}}.c:1

        .text
main:
        .file   1 "w/e/r" "t.c"
        .loc    1 1 0                   # w/e/r/t.c:1:0
        retq

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   0                       # DW_CHILDREN_no
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   16                      # DW_AT_stmt_list
        .byte   23                      # DW_FORM_sec_offset
        .byte   27                      # DW_AT_comp_dir
        .byte   8                       # DW_FORM_string
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
        .byte   1                       # Abbrev [1] 0xb:0x40 DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .asciz  "w/e/r/t.c"             # DW_AT_name
        .long   .Lline_table_start0     # DW_AT_stmt_list
        .asciz  "q"                     # DW_AT_comp_dir
.Ldebug_info_end0:

        .section        .debug_line,"",@progbits
.Lline_table_start0:
