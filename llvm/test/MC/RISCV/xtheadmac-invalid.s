# RUN: llvm-mc --verify=expected,rv32 -triple riscv32 -mattr=+xtheadmac < %s
# RUN: llvm-mc --verify=expected,rv64 -triple riscv64 -mattr=+xtheadmac < %s

th.mula  t0, t1, 0(t2) # expected-error:18 {{register must be a GPR}}

th.muls  t0, 12, t2 # expected-error:14 {{register must be a GPR}}

th.mulsh t0, t1, 34 # expected-error:18 {{register must be a GPR}}

th.mulsw t0, -56, t2    
# rv64-error@-1:14 {{register must be a GPR}}
# rv32-error@-2:1 {{invalid instruction}}
