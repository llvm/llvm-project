@ RUN: llvm-mc -triple=thumbv7-apple-darwin -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=thumbv6-apple-darwin -mcpu=cortex-m0 -show-encoding < %s | FileCheck %s 
@ RUN: not llvm-mc -triple=thumbv6-apple-darwin -show-encoding < %s > %t 2> %t2
@ RUN: FileCheck %s --check-prefix=CHECK-EVIL-PRE-UAL < %t
@ RUN: FileCheck %s --check-prefix CHECK-ERROR < %t2

  .syntax unified

        nop
        yield
        wfe
        wfi
        sev
@ CHECK: nop                            @ encoding: [0x00,0xbf]
@ CHECK: yield                          @ encoding: [0x10,0xbf]
@ CHECK: wfe                            @ encoding: [0x20,0xbf]
@ CHECK: wfi                            @ encoding: [0x30,0xbf]
@ CHECK: sev                            @ encoding: [0x40,0xbf]

@ CHECK-EVIL-PRE-UAL: mov r8, r8                     @ encoding: [0xc0,0x46]

        dmb sy
        dmb
        dsb sy
        dsb
        isb sy
        isb
@ CHECK: dmb	sy                      @ encoding: [0xbf,0xf3,0x5f,0x8f]
@ CHECK: dmb	sy                      @ encoding: [0xbf,0xf3,0x5f,0x8f]
@ CHECK: dsb	sy                      @ encoding: [0xbf,0xf3,0x4f,0x8f]
@ CHECK: dsb	sy                      @ encoding: [0xbf,0xf3,0x4f,0x8f]
@ CHECK: isb	sy                      @ encoding: [0xbf,0xf3,0x6f,0x8f]
@ CHECK: isb	sy                      @ encoding: [0xbf,0xf3,0x6f,0x8f]


@ CHECK-ERROR: error: instruction requires: armv6m or armv6t2
@ CHECK-ERROR-NEXT: yield

@ CHECK-ERROR: error: instruction requires: armv6m or armv6t2
@ CHECK-ERROR-NEXT: wfe

@ CHECK-ERROR: error: instruction requires: armv6m or armv6t2
@ CHECK-ERROR-NEXT: wfi

@ CHECK-ERROR: error: instruction requires: armv6m or armv6t2
@ CHECK-ERROR-NEXT: sev

@ CHECK-ERROR: error:
@ CHECK-ERROR-NEXT: dmb sy

@ CHECK-ERROR: error: instruction requires: data-barriers
@ CHECK-ERROR-NEXT: dmb

@ CHECK-ERROR: error:
@ CHECK-ERROR-NEXT: dsb sy

@ CHECK-ERROR: error: instruction requires: data-barriers
@ CHECK-ERROR-NEXT: dsb

@ CHECK-ERROR: error:
@ CHECK-ERROR-NEXT: isb sy

@ CHECK-ERROR: error: instruction requires: data-barriers
@ CHECK-ERROR-NEXT: isb
