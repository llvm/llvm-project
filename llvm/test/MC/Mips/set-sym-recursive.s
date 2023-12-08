# RUN: not llvm-mc -triple mips-unknown-linux %s 2>&1 | FileCheck %s

.set A, A + 1
# CHECK: :[[@LINE-1]]:9: error: Recursive use of 'A'
.word A
