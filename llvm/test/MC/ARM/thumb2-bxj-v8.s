@ RUN: not llvm-mc -triple=thumbv6t2--none-eabi -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=UNDEF
@ RUN: not llvm-mc -triple=thumbv7a--none-eabi -show-encoding < %s 2>&1  | FileCheck %s --check-prefix=UNDEF
@ RUN: not llvm-mc -triple=thumbv7r--none-eabi -show-encoding < %s 2>&1  | FileCheck %s --check-prefix=UNDEF
@ RUN: not llvm-mc -triple=thumbv7m--none-eabi -show-encoding < %s 2>&1  | FileCheck %s --check-prefix=ARM_MODE
@ RUN: llvm-mc -triple=thumbv8a--none-eabi -show-encoding < %s 2>&1  | FileCheck %s

bxj r13

@ CHECK: bxj	sp                      @ encoding: [0xcd,0xf3,0x00,0x8f]
@ UNDEF:  error: r13 (SP) is an unpredictable operand to BXJ
@ ARM_MODE: error: instruction requires: !armv*m
