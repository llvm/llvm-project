# RUN: not llvm-mc %s -triple=riscv32 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple=riscv64 2>&1 | FileCheck %s

lga  x1, %tlsdesc_hi(1234) # CHECK: :[[@LINE]]:10: error: operand must be a bare symbol name
lga  x1, %tlsdesc_hi(foo) # CHECK: :[[@LINE]]:10: error: operand must be a bare symbol name

lw   a0, t0, %tlsdesc_load_lo(a_symbol) # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
lw   a0, t0, %tlsdesc_load_lo(a_symbol)(a4) # CHECK: :[[@LINE]]:14: error: invalid operand for instruction

addi a0, t0, %tlsdesc_add_lo(a_symbol)(a4) # CHECK: :[[@LINE]]:40: error: invalid operand for instruction
addi a0, %tlsdesc_add_lo(a_symbol) # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

jalr t0, a0, %tlsdesc_call(a_symbol) # CHECK: :[[@LINE]]:14: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]
jalr t0, 12345(a1), %tlsdesc_call(a_symbol) # CHECK: :[[@LINE]]:10: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]
