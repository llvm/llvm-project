# A function whose entry point (low_pc) is not covered by any line table row.
# The first instruction is emitted before the first .loc, so the first line
# table entry begins at low_pc+1, leaving the entry address uncovered. This
# mirrors the WebAssembly case where the function entry points at the locals
# declaration (which has no line entry) rather than at an executable
# instruction.
#
# Function::GetPrologueByteSize() must fall back to the first line entry that
# begins within the function and skip the prologue to a real instruction,
# instead of leaving the breakpoint sitting on the (unexecutable) entry
# address.

# RUN: split-file %s %t
# RUN: llvm-mc -triple x86_64-pc-linux -filetype=obj %t/input.s -o %t/input.o
# RUN: %lldb %t/input.o -s %t/commands -o exit | FileCheck %s

#--- commands

# With prologue skipping (the default), the breakpoint must move past the
# uncovered entry address (0x0) to the prologue_end line entry at 0x2.
breakpoint set --name foo
# CHECK-LABEL: breakpoint set --name foo
# CHECK: Breakpoint 1: where = input.o`foo + 2 at {{.*}}:2, address = 0x0000000000000002

# Without prologue skipping the breakpoint sits on the function entry (0x0).
# The entry address is not covered by a line entry, so no "at <file>:<line>"
# location is printed.
breakpoint set --name foo --skip-prologue false
# CHECK-LABEL: breakpoint set --name foo --skip-prologue false
# CHECK: Breakpoint 2: where = input.o`foo, address = 0x0000000000000000

#--- input.s
        .text
        .file   0 "." "-"

        .type   foo,@function
foo:
        # No .loc here: the entry instruction is deliberately left uncovered by
        # the line table, so the first line entry begins at low_pc + 1.
        nop
        .loc    0 1
        nop
        .loc    0 2 prologue_end
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
