# RUN: llvm-mc --verify -triple riscv32 -mattr=+xtheadcondmov < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+xtheadcondmov < %s

th.mveqz a0,a1        # expected-error:15 {{too few operands for instruction}}
th.mveqz a0,a1,a2,a3  # expected-error:19 {{unexpected extra operand for instruction}}
th.mveqz a0,a1,1      # expected-error:16 {{register must be a GPR}}
th.mvnez a0,a1        # expected-error:15 {{too few operands for instruction}}
th.mvnez a0,a1,a2,a3  # expected-error:19 {{unexpected extra operand for instruction}}
th.mvnez a0,a1,1      # expected-error:16 {{register must be a GPR}}
