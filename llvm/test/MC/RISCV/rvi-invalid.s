# RUN: not llvm-mc -triple=riscv32 < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=riscv64 < %s 2>&1 | FileCheck %s

reldef:

.global undef

## simm12
addi a0, a1, reldef   # CHECK: :[[@LINE]]:14: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
addi a0, a1, reldef-. # CHECK: :[[@LINE]]:14: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
addi a0, a1, undef    # CHECK: :[[@LINE]]:14: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
addi a0, a1, latedef  # CHECK: :[[@LINE]]:14: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]


lui a0, reldef   # CHECK: :[[@LINE]]:9: error: operand must be a symbol with %hi/%tprel_hi specifier or an integer in the range [0, 1048575]
lui a0, reldef-. # CHECK: :[[@LINE]]:9: error: operand must be a symbol with %hi/%tprel_hi specifier or an integer in the range [0, 1048575]
lui a0, undef    # CHECK: :[[@LINE]]:9: error: operand must be a symbol with %hi/%tprel_hi specifier or an integer in the range [0, 1048575]
lui a0, latedef  # CHECK: :[[@LINE]]:9: error: operand must be a symbol with %hi/%tprel_hi specifier or an integer in the range [0, 1048575]

auipc a0, reldef   # CHECK: :[[@LINE]]:11: error: operand must be a symbol with a %pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi specifier or an integer in the range [0, 1048575]
auipc a0, reldef-. # CHECK: :[[@LINE]]:11: error: operand must be a symbol with a %pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi specifier or an integer in the range [0, 1048575]
auipc a0, undef    # CHECK: :[[@LINE]]:11: error: operand must be a symbol with a %pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi specifier or an integer in the range [0, 1048575]
auipc a0, latedef  # CHECK: :[[@LINE]]:11: error: operand must be a symbol with a %pcrel_hi/%got_pcrel_hi/%tls_ie_pcrel_hi/%tls_gd_pcrel_hi specifier or an integer in the range [0, 1048575]

.set latedef, 1
