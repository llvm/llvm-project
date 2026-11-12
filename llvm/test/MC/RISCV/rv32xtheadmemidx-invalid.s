# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadmemidx < %s 2>&1 | FileCheck --match-full-lines %s

th.lwuia	a0, (a1), 0, 0    # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
th.lwuib	a0, (a1), 15, 1   # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
th.ldia		a0, (a1), 0, 0    # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
th.ldib		a0, (a1), 0, 0    # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
th.sdia		a0, (a1), -16, 0  # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
th.sdib		a0, (a1), -1, 1   # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
th.lrwu		a0, a1, a2, 2     # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
th.lrd		a0, a1, a2, 0     # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
th.srd		a0, a1, a2, 3     # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
th.lurwu	a0, a1, a2, 2     # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
th.lurd		a0, a1, a2, 0     # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
th.surd		a0, a1, a2, 3     # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set