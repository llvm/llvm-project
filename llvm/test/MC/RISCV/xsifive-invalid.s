# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck %s

cflush.d.l1 x0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSiFivecflushdlone' (SiFive cflush.d.l1 Instruction){{$}}

cflush.d.l1 x7 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSiFivecflushdlone' (SiFive cflush.d.l1 Instruction){{$}}

cdiscard.d.l1 x0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSiFivecdiscarddlone' (SiFive cdiscard.d.l1 Instruction){{$}}

cdiscard.d.l1 x7 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSiFivecdiscarddlone' (SiFive cdiscard.d.l1 Instruction){{$}}
