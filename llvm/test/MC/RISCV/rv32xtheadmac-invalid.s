# RUN: llvm-mc --verify -triple riscv32 -mattr=+xtheadmac < %s

th.mulaw  t0, t1, t2     # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
th.mulsw  t0, t1, t2     # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
