# RUN: llvm-mc --verify -triple riscv32 -mattr=+d -mattr=+xtheadfmemidx < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+d -mattr=+xtheadfmemidx < %s

th.flrd fa0, a1, a2, 5     # expected-error:22 {{immediate must be an integer in the range [0, 3]}}
th.flrd a0, a1, a2, 3      # expected-error:9 {{register must be a FPR}}
th.flrw 0(fa0), a1, a2, 0  # expected-error:9 {{register must be a FPR}}
th.flrw fa0, 4(a1), a2, 3  # expected-error:14 {{register must be a GPR}}
th.fsrd fa0, a1, -1(a2), 0 # expected-error:18 {{register must be a GPR}}
th.fsrd fa0, a1, a2, -3    # expected-error:22 {{immediate must be an integer in the range [0, 3]}}
