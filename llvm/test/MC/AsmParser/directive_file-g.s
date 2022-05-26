## Make sure that using -g (or equivalent) on an asm file that already has
## debug-info directives in it will correctly ignore the -g and produce
## debug info corresponding to the directives in the source.
## Note gcc accepts ".file 1" after a label, although not after an opcode.
## If no other directives appear, gcc emits no debug info at all.

# RUN: llvm-mc -g -triple i386-unknown-unknown -filetype=obj %s -o %t
# RUN: llvm-dwarfdump -debug-info -debug-line %t | FileCheck %s

foo:
        .file 1 "a.c"
        .loc 1 1 1
        nop

# CHECK: .debug_info
## gcc does generate a DW_TAG_compile_unit in this case, with or without
## -g on the command line, but we do not.
# CHECK-EMPTY:
# CHECK-NEXT: .debug_line
# CHECK: file_names[ 1]:
# CHECK-NEXT: name: "a.c"
# CHECK-NEXT: dir_index: 0
# CHECK: 0x{{0+}}0 1 1 1 0 0 is_stmt
# CHECK: 0x{{0+}}1 1 1 1 0 0 is_stmt end_sequence
