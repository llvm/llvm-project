@ RUN: llvm-mc -triple thumbv7-eabi -mattr=+virtualization -show-encoding %s | FileCheck %s --check-prefix=CHECK
@ RUN: not llvm-mc -triple thumbv7-eabi -mcpu=cortex-a9 %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOVIRT

        .syntax unified
        .text
        .thumb

        hvc     #0
        hvc.w   #0

@ CHECK: hvc.w #0                      @ encoding: [0xe0,0xf7,0x00,0x80]
@ CHECK: hvc.w #0                      @ encoding: [0xe0,0xf7,0x00,0x80]

@ CHECK-NOVIRT: error: instruction requires: virtualization-extensions
@ CHECK-NOVIRT: hvc     #0
@ CHECK-NOVIRT: ^
@ CHECK-NOVIRT: error: instruction requires: virtualization-extensions
@ CHECK-NOVIRT: hvc.w   #0
@ CHECK-NOVIRT: ^
