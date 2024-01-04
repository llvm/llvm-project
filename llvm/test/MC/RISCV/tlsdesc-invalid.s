# RUN: not llvm-mc %s -triple=riscv32 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple=riscv64 2>&1 | FileCheck %s

auipc x1, %tlsdesc_call(foo) # CHECK: :[[@LINE]]:11: error: operand must be a symbol with a %pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi modifier or an integer in the range
auipc x1, %tlsdesc_call(1234) # CHECK: :[[@LINE]]:11: error: operand must be a symbol with a %pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi modifier or an integer in the range

lw   a0, t0, %tlsdesc_load_lo(a_symbol) # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
lw   a0, t0, %tlsdesc_load_lo(a_symbol)(a4) # CHECK: :[[@LINE]]:14: error: invalid operand for instruction

addi a0, t0, %tlsdesc_add_lo(a_symbol)(a4) # CHECK: :[[@LINE]]:40: error: invalid operand for instruction
addi a0, %tlsdesc_add_lo(a_symbol) # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

jalr x5, 0(a1), %tlsdesc_hi(a_symbol) # CHECK: :[[@LINE]]:17: error: operand must be a symbol with %tlsdesc_call modifier
jalr x1, 0(a1), %tlsdesc_call(a_symbol) # CHECK: :[[@LINE]]:12: error: the output operand must be t0/x5 when using %tlsdesc_call modifier
