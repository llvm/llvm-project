# RUN: not llvm-mc -triple riscv32 -mattr=+f -mattr=+xtheadfmemidx < %s 2>&1 | FileCheck %s

th.flrd fa0, a1, a2, 0  # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
th.fsrd fa0, a1, a2, 0  # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'D' (Double-Precision Floating-Point){{$}}
th.flurd fa0, a1, a2, 0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'D' (Double-Precision Floating-Point), RV64I Base Instruction Set{{$}}
th.flurw fa0, a1, a2, 0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
th.fsurd fa0, a1, a2, 0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'D' (Double-Precision Floating-Point), RV64I Base Instruction Set{{$}}
th.fsurw fa0, a1, a2, 0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
