# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck --match-full-lines %s
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck --match-full-lines %s

sf.cflush.d.l1 0x10 # CHECK: <stdin>:[[@LINE]]:1: error: invalid instruction

sf.cdiscard.d.l1 0x10 # CHECK: <stdin>:[[@LINE]]:1: error: invalid instruction

sf.cflush.d.l1 x0 # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'XSiFivecflushdlone' (SiFive sf.cflush.d.l1 Instruction)

sf.cflush.d.l1 x7 # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'XSiFivecflushdlone' (SiFive sf.cflush.d.l1 Instruction)

sf.cdiscard.d.l1 x0 # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'XSiFivecdiscarddlone' (SiFive sf.cdiscard.d.l1 Instruction)

sf.cdiscard.d.l1 x7 # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'XSiFivecdiscarddlone' (SiFive sf.cdiscard.d.l1 Instruction)

sf.cease x1 # CHECK: <stdin>:[[@LINE]]:1: error: invalid instruction

sf.cease 0x10 # CHECK: <stdin>:[[@LINE]]:1: error: invalid instruction

sf.cease # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'XSfcease' (SiFive sf.cease Instruction)
