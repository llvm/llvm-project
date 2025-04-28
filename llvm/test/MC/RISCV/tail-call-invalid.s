# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck %s

tail 1234 # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
tail %pcrel_hi(1234) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
tail %pcrel_lo(1234) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
tail %pcrel_hi(foo) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
tail %pcrel_lo(foo) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
tail %hi(1234) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
tail %lo(1234) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
tail %hi(foo) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
tail %lo(foo) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
