# A function whose entry point (low_pc) is not covered by any line table row:
# the first instruction is emitted before the first .loc, so the first line
# entry begins at low_pc+1. This mirrors WebAssembly, where a function begins
# with a locals declaration that carries no line information.
#
# Function::GetStartLineEntry must fall back to the first line entry within the
# function so "source list --name" can find the function's source instead of
# reporting no line information.

# RUN: split-file %s %t
# RUN: llvm-mc -triple x86_64-pc-linux -filetype=obj %t/input.s -o %t/input.o
# RUN: cd %t && %lldb input.o -o "source list --name foo" -o exit | FileCheck %s

# CHECK-LABEL: source list --name foo
# CHECK: File: inlined.c
# CHECK: void foo()
# CHECK: source list marker

#--- inlined.c
void stop();
void foo() {
  // This is the source list marker.
  stop();
}

#--- input.s
        .text
        .file   0 "." "inlined.c"

        .type   foo,@function
foo:
        # No .loc here: the entry instruction is deliberately left uncovered by
        # the line table, so the first line entry begins at low_pc + 1.
        nop
        .loc    0 2
        nop
        .loc    0 4 prologue_end
        nop
        retq
.Lfoo_end:
        .size   foo, .Lfoo_end-foo

        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   8                               # DW_FORM_string
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   1                               # DW_FORM_addr
        .byte   16                              # DW_AT_stmt_list
        .byte   23                              # DW_FORM_sec_offset
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   2                               # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   0                               # DW_CHILDREN_no
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   1                               # DW_FORM_addr
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
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
        .short  29                              # DW_AT_language
        .quad   foo                             # DW_AT_low_pc
        .quad   .Lfoo_end                       # DW_AT_high_pc
        .long   .Lline_table_start0             # DW_AT_stmt_list
        .byte   2                               # Abbrev [2] DW_TAG_subprogram
        .quad   foo                             # DW_AT_low_pc
        .quad   .Lfoo_end                       # DW_AT_high_pc
        .asciz  "foo"                           # DW_AT_name
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:

        .section        ".note.GNU-stack","",@progbits
        .section        .debug_line,"",@progbits
.Lline_table_start0:
