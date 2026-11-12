@ RUN: llvm-mc -triple=thumbv6t2--none-eabi -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=thumbv7a--none-eabi -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=thumbv7r--none-eabi -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=thumbv8a--none-eabi -show-encoding < %s | FileCheck %s
@ RUN: not llvm-mc -triple=thumbv7m--none-eabi -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=UNDEF

        bxj r2

@ CHECK: bxj r2                      @ encoding: [0xc2,0xf3,0x00,0x8f]
@ UNDEF: error: instruction requires: !armv*m
