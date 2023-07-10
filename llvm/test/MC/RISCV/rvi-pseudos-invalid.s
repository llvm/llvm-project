# RUN: not llvm-mc %s -triple=riscv32 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple=riscv64 2>&1 | FileCheck %s

lga x1, 1234 # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lga x1, %pcrel_hi(1234) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lga x1, %pcrel_lo(1234) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lga x1, %pcrel_hi(foo) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lga x1, %pcrel_lo(foo) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lga x1, %hi(1234) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lga x1, %lo(1234) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lga x1, %hi(foo) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name
lga x1, %lo(foo) # CHECK: :[[@LINE]]:9: error: operand must be a bare symbol name

sw a2, %hi(a_symbol), a3 # CHECK: :[[@LINE]]:8: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]
sw a2, %lo(a_symbol), a3 # CHECK: :[[@LINE]]:23: error: invalid operand for instruction
sw a2, %lo(a_symbol)(a4), a3 # CHECK: :[[@LINE]]:27: error: invalid operand for instruction

# Too few operands must be rejected
sw a2, a_symbol # CHECK: :[[@LINE]]:1: error: too few operands for instruction
