# RUN: not llvm-mc -triple riscv32 -mattr=+d -mattr=+xtheadfmemidx < %s 2>&1 | FileCheck --match-full-lines %s
# RUN: not llvm-mc -triple riscv64 -mattr=+d -mattr=+xtheadfmemidx < %s 2>&1 | FileCheck --match-full-lines %s

th.flrd fa0, a1, a2, 5     # CHECK: <stdin>:[[@LINE]]:22: error: immediate must be an integer in the range [0, 3]
th.flrd a0, a1, a2, 3      # CHECK: <stdin>:[[@LINE]]:9: error: register must be a FPR
th.flrw 0(fa0), a1, a2, 0  # CHECK: <stdin>:[[@LINE]]:9: error: register must be a FPR
th.flrw fa0, 4(a1), a2, 3  # CHECK: <stdin>:[[@LINE]]:14: error: register must be a GPR
th.fsrd fa0, a1, -1(a2), 0 # CHECK: <stdin>:[[@LINE]]:18: error: register must be a GPR
th.fsrd fa0, a1, a2, -3    # CHECK: <stdin>:[[@LINE]]:22: error: immediate must be an integer in the range [0, 3]
