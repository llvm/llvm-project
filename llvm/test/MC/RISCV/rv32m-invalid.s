# RUN: llvm-mc --verify -triple riscv32 -mattr=+m < %s

# RV64M instructions can't be used for RV32
mulw ra, sp, gp # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
divw tp, t0, t1 # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
divuw t2, s0, s2 # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
remw a0, a1, a2 # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
remuw a3, a4, a5 # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

