# RUN: not llvm-mc %s -triple=riscv64 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple=riscv64 2>&1 | FileCheck %s

lla x1, 1234 # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lla x1, %pcrel_hi(1234) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lla x1, %pcrel_lo(1234) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lla x1, %pcrel_hi(foo) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lla x1, %pcrel_lo(foo) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lla x1, %hi(1234) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lla x1, %lo(1234) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lla x1, %hi(foo) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lla x1, %lo(foo) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name

la x1, 1234 # CHECK: :[[@LINE]]:8: error: operand must be a bare symbol name
la x1, %pcrel_hi(1234) # CHECK: :[[@LINE]]:8: error: operand must be a bare symbol name
la x1, %pcrel_lo(1234) # CHECK: :[[@LINE]]:8: error: operand must be a bare symbol name
la x1, %pcrel_hi(foo) # CHECK: :[[@LINE]]:8: error: operand must be a bare symbol name
la x1, %pcrel_lo(foo) # CHECK: :[[@LINE]]:8: error: operand must be a bare symbol name
la x1, %hi(1234) # CHECK: :[[@LINE]]:8: error: operand must be a bare symbol name
la x1, %lo(1234) # CHECK: :[[@LINE]]:8: error: operand must be a bare symbol name
la x1, %hi(foo) # CHECK: :[[@LINE]]:8: error: operand must be a bare symbol name
la x1, %lo(foo) # CHECK: :[[@LINE]]:8: error: operand must be a bare symbol name
