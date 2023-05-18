@ RUN: llvm-mc -triple=thumbv6t2--none-eabi -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=thumbv7a--none-eabi -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=thumbv7r--none-eabi -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=thumbv8a--none-eabi -show-encoding < %s | FileCheck %s
@ RUN: not llvm-mc -triple=thumbv7m--none-eabi -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=UNDEF

  ldrexd r0, r1, [r2]
  strexd r3, r4, r5, [r6]

@ CHECK: ldrexd r0, r1, [r2]            @ encoding: [0xd2,0xe8,0x7f,0x01]
@ CHECK: strexd r3, r4, r5, [r6]        @ encoding: [0xc6,0xe8,0x73,0x45]

@ UNDEF: error: instruction requires: !armv*m
@ UNDEF: error: instruction requires: !armv*m
