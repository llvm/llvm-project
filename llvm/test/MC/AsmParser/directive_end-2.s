# RUN: llvm-mc -triple i386-unknown-unknown %s -I %p -filetype obj -o - \
# RUN:   | llvm-readobj --symbols - | FileCheck %s

rock:
    movl $42, %eax

.include "directive_end.s"

hard_place:
    movl $42, %ebx

# CHECK: Symbol {
# CHECK:   Name: rock
# CHECK-NOT:   Name: hard_place
