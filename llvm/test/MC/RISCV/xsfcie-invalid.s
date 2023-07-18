# SCIE - SiFive Custom Instructions Extension.
# RUN: not llvm-mc -triple riscv32 -mattr=-xsfcie < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=-xsfcie < %s 2>&1 | FileCheck %s

cflush.d.l1 0x10 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction

cdiscard.d.l1 0x10 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction

cflush.d.l1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSfcie' (SiFive Custom Instruction Extension SCIE.)

cdiscard.d.l1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSfcie' (SiFive Custom Instruction Extension SCIE.)

cflush.d.l1 x0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSfcie' (SiFive Custom Instruction Extension SCIE.)

cflush.d.l1 x7 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSfcie' (SiFive Custom Instruction Extension SCIE.)

cdiscard.d.l1 x0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSfcie' (SiFive Custom Instruction Extension SCIE.)

cdiscard.d.l1 x7 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSfcie' (SiFive Custom Instruction Extension SCIE.)

cease x1 # CHECK: :[[@LINE]]:7: error: invalid operand for instruction

cease 0x10 # CHECK: :[[@LINE]]:7: error: invalid operand for instruction

cease # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'XSfcie' (SiFive Custom Instruction Extension SCIE.)

csrr t1, mbpm # CHECK: :[[@LINE]]:10: error: system register use requires an option to be enabled

csrr t1, mfd # CHECK: :[[@LINE]]:10: error: system register use requires an option to be enabled

csrr t1, mpd # CHECK: :[[@LINE]]:10: error: system register use requires an option to be enabled

csrr t1, mnscratch # CHECK: :[[@LINE]]:10: error: system register use requires an option to be enabled

csrr t1, mnepc # CHECK: :[[@LINE]]:10: error: system register use requires an option to be enabled

csrr t1, mncause # CHECK: :[[@LINE]]:10: error: system register use requires an option to be enabled

csrr t1, mnstatus # CHECK: :[[@LINE]]:10: error: system register use requires an option to be enabled
