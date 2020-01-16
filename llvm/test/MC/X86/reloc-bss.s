# RUN: not llvm-mc -filetype=obj -triple=x86_64-linux-gnu %s 2>&1 | FileCheck %s
# CHECK: LLVM ERROR: cannot have fixups in virtual section!

.section        .init_array,"awT",@nobits

.hidden patatino
.globl  patatino
patatino:
  movl __init_array_start, %eax
