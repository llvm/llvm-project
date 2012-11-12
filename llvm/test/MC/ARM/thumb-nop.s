@ RUN: llvm-mc -triple=thumbv6-apple-darwin -show-encoding < %s | FileCheck %s -check-prefix=CHECK-V6
@ RUN: llvm-mc -triple=thumbv7-apple-darwin -show-encoding < %s | FileCheck %s -check-prefix=CHECK-V7

  .syntax unified

        nop

@ CHECK-V6: nop                            @ encoding: [0xc0,0x46]
@ CHECK-V7: nop                            @ encoding: [0x00,0xbf]
