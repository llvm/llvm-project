# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadmac < %s 2>&1 | FileCheck --match-full-lines %s

th.mulaw  t0, t1, t2     # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
th.mulsw  t0, t1, t2     # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
