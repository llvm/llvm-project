# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o - | llvm-dwarfdump --debug-line - | FileCheck %s

# The line table is open in the MC path.
# The end sequence is emitted using the section end label.

# CHECK: 0x0000000000000001 [[T:.*]] end_sequence
# CHECK: 0x0000000000000001 [[T:.*]] end_sequence

        .text
        .section        .text.f1
f1:
        .file   1 "/" "t1.c"
        .loc    1 1 0
        nop

        .section        .text.f2
f2:
        .loc    1 2 0
        nop
