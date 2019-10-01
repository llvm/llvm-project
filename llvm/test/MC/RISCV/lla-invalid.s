# RUN: not llvm-mc -triple=riscv32 < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=riscv64 < %s 2>&1 | FileCheck %s

# Non bare symbols must be rejected
lla a2, %lo(a_symbol) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lla a2, %hi(a_symbol) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lla a2, foo@plt # CHECK: :[[@LINE]]:17: error: '@plt' operand not valid for instruction
