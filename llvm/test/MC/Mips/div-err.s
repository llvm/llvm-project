# RUN: not llvm-mc %s -triple=mips -mcpu=mips32 2>&1 | FileCheck %s

# CHECK: error: expected immediate operand kind
  div $t0, $t1, f0
