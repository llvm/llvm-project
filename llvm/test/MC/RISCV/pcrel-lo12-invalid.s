# RUN: not llvm-mc -triple riscv32 -mattr=-relax -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv32 -mattr=+relax -filetype obj < %s -o /dev/null 2>&1 | FileCheck %s

1:
  addi a0, a0, %pcrel_lo(1b) # CHECK: :[[@LINE]]:3: error: could not find corresponding %pcrel_hi
