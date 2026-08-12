# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadmemidx < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-RV32 %s
# RUN: not llvm-mc -triple riscv64 -mattr=+xtheadmemidx < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-RV64 %s

# CHECK-RV32: :[[@LINE+2]]:1: error: instruction requires the following: RV64I Base Instruction Set
# CHECK-RV64: :[[@LINE+1]]:9: error: register must be a GPR
th.ldia 0(a0), (a1), 0, 0
# CHECK-RV32: :[[@LINE+2]]:1: error: instruction requires the following: RV64I Base Instruction Set
# CHECK-RV64: :[[@LINE+1]]:14: error: expected '('
th.ldib		a0, 2(a1), 15, 1
# CHECK-RV32: :[[@LINE+2]]:20: error: immediate must be an integer in the range [-16, 15]
# CHECK-RV64: :[[@LINE+1]]:20: error: immediate must be an integer in the range [-16, 15]
th.lwia		a0, (a1), 30, 2
# CHECK-RV32: :[[@LINE+2]]:25: error: immediate must be an integer in the range [0, 3]
# CHECK-RV64: :[[@LINE+1]]:25: error: immediate must be an integer in the range [0, 3]
th.lwib		a0, (a1), -16, 43
# CHECK-RV32: :[[@LINE+2]]:20: error: immediate must be an integer in the range [-16, 15]
# CHECK-RV64: :[[@LINE+1]]:20: error: immediate must be an integer in the range [-16, 15]
th.lhib		a0, (a1), -17, 3
# CHECK-RV32: :[[@LINE+2]]:9: error: register must be a GPR
# CHECK-RV64: :[[@LINE+1]]:9: error: register must be a GPR
th.lrb		-2(a0), a1, a2, 0
# CHECK-RV32: :[[@LINE+2]]:13: error: register must be a GPR
# CHECK-RV64: :[[@LINE+1]]:13: error: register must be a GPR
th.lrw		a0, 3(a1), a2, 1
# CHECK-RV32: :[[@LINE+2]]:17: error: register must be a GPR
# CHECK-RV64: :[[@LINE+1]]:17: error: register must be a GPR
th.lrw		a0, a1, 4(a2), 2
# CHECK-RV32: :[[@LINE+2]]:21: error: immediate must be an integer in the range [0, 3]
# CHECK-RV64: :[[@LINE+1]]:21: error: immediate must be an integer in the range [0, 3]
th.lrh		a0, a1, a2, 5
# CHECK-RV32: :[[@LINE+2]]:22: error: immediate must be an integer in the range [0, 3]
# CHECK-RV64: :[[@LINE+1]]:22: error: immediate must be an integer in the range [0, 3]
th.lrhu		a0, a1, a2, -1
