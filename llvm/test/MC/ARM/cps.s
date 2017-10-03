@ RUN: llvm-mc -triple=thumbv6t2--none-eabi -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=thumbv7a--none-eabi -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=thumbv7r--none-eabi -show-encoding < %s | FileCheck %s
@ RUN: llvm-mc -triple=thumbv8a--none-eabi -show-encoding < %s | FileCheck %s
@ RUN: not llvm-mc -triple=thumbv7m--none-eabi -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=UNDEF

  cpsie f
  cpsie i, #3
  cps #0

@ CHECK: cpsie f                         @ encoding: [0x61,0xb6]
@ CHECK: cpsie   i, #3                   @ encoding: [0xaf,0xf3,0x43,0x85]
@ CHECK: cps     #0                      @ encoding: [0xaf,0xf3,0x00,0x81]

@ UNDEF-DAG: cpsie f                         @ encoding: [0x61,0xb6]
@ UNDEF-DAG: instruction requires:
@ UNDEF-DAG: error: instruction 'cps' requires effect for M-class
