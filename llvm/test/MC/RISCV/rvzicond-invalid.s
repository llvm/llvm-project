# RUN: llvm-mc --verify -triple riscv32 -mattr=+zicond < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+zicond < %s

# Use of operand modifier on register name
czero.eqz t1, %lo(t2), t3 # expected-error:15 {{register must be a GPR}}

# Invalid register name
czero.nez a4, a3, foo # expected-error:19 {{register must be a GPR}}

# Invalid operand type
czero.eqz t1, 2, t3 # expected-error:15 {{register must be a GPR}}

# Too many operands
czero.eqz t1, t2, t3, t4 # expected-error:23 {{unexpected extra operand for instruction}}
czero.nez t1, t2, t3, 4 # expected-error:23 {{unexpected extra operand for instruction}}

# Too few operands
czero.eqz t1, t2 # expected-error:17 {{too few operands for instruction}}
