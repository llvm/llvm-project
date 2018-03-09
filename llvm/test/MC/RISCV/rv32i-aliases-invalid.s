# RUN: not llvm-mc %s -triple=riscv32 -riscv-no-aliases 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple=riscv32 2>&1 | FileCheck %s

# TODO ld
# TODO sd

negw x1, x2   # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled
sext.w x3, x4 # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled
