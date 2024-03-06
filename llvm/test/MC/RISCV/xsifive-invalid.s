# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck %s

sf.cflush.d.l1 x0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSiFivecflushdlone' (SiFive sf.cflush.d.l1 Instruction){{$}}

sf.cflush.d.l1 x7 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSiFivecflushdlone' (SiFive sf.cflush.d.l1 Instruction){{$}}

sf.cdiscard.d.l1 x0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSiFivecdiscarddlone' (SiFive sf.cdiscard.d.l1 Instruction){{$}}

sf.cdiscard.d.l1 x7 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSiFivecdiscarddlone' (SiFive sf.cdiscard.d.l1 Instruction){{$}}
