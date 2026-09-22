# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadmemidx < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,RV32
# RUN: not llvm-mc -triple riscv64 -mattr=+xtheadmemidx < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,RV64

th.ldia		0(a0), (a1), 0, 0  # CHECK: :[[@LINE]]:10: error: register must be a GPR
th.ldib		a0, 2(a1), 15, 1   # CHECK: :[[@LINE]]:14: error: expected '('
th.lwia		a0, (a1), 30, 2    # CHECK: :[[@LINE]]:20: error: immediate must be an integer in the range [-16, 15]
th.lwib		a0, (a1), -16, 43  # CHECK: :[[@LINE]]:25: error: immediate must be an integer in the range [0, 3]
th.lhib		a0, (a1), -17, 3   # CHECK: :[[@LINE]]:20: error: immediate must be an integer in the range [-16, 15]
th.lrb		-2(a0), a1, a2, 0  # CHECK: :[[@LINE]]:9: error: register must be a GPR
th.lrw		a0, 3(a1), a2, 1   # CHECK: :[[@LINE]]:13: error: register must be a GPR
th.lrw		a0, a1, 4(a2), 2   # CHECK: :[[@LINE]]:17: error: register must be a GPR
th.lrh		a0, a1, a2, 5      # CHECK: :[[@LINE]]:21: error: immediate must be an integer in the range [0, 3]
th.lrhu		a0, a1, a2, -1     # CHECK: :[[@LINE]]:22: error: immediate must be an integer in the range [0, 3]

th.lbia  a0, (a0), 0, 0     # CHECK: :[[@LINE]]:10: error: rd and rs1 must be different
th.lbib  a0, (a0), 0, 0     # CHECK: :[[@LINE]]:10: error: rd and rs1 must be different
th.lbuia a0, (a0), 0, 0     # CHECK: :[[@LINE]]:10: error: rd and rs1 must be different
th.lbuib a0, (a0), 0, 0     # CHECK: :[[@LINE]]:10: error: rd and rs1 must be different
th.lhia  a0, (a0), 0, 0     # CHECK: :[[@LINE]]:10: error: rd and rs1 must be different
th.lhib  a0, (a0), 0, 0     # CHECK: :[[@LINE]]:10: error: rd and rs1 must be different
th.lhuia a0, (a0), 0, 0     # CHECK: :[[@LINE]]:10: error: rd and rs1 must be different
th.lhuib a0, (a0), 0, 0     # CHECK: :[[@LINE]]:10: error: rd and rs1 must be different
th.lwia  a0, (a0), 0, 0     # CHECK: :[[@LINE]]:10: error: rd and rs1 must be different
th.lwib  a0, (a0), 0, 0     # CHECK: :[[@LINE]]:10: error: rd and rs1 must be different
th.lwuia a0, (a0), 0, 0     # RV32: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
                            # RV64: :[[@LINE-1]]:10: error: rd and rs1 must be different
th.lwuib a0, (a0), 0, 0     # RV32: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
                            # RV64: :[[@LINE-1]]:10: error: rd and rs1 must be different
th.ldia  a0, (a0), 0, 0     # RV32: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
                            # RV64: :[[@LINE-1]]:10: error: rd and rs1 must be different
th.ldib  a0, (a0), 0, 0     # RV32: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
                            # RV64: :[[@LINE-1]]:10: error: rd and rs1 must be different
th.lbia  a0, (x10), 1, 0    # CHECK: :[[@LINE]]:10: error: rd and rs1 must be different
th.lbia  zero, (zero), 0, 0 # CHECK: :[[@LINE]]:10: error: rd and rs1 must be different
