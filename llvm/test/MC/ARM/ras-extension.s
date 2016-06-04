@ RUN: llvm-mc -triple   armv8a-none-eabi -mattr=+ras -show-encoding %s | FileCheck %s --check-prefix=ARM
@ RUN: llvm-mc -triple thumbv8a-none-eabi -mattr=+ras -show-encoding %s | FileCheck %s --check-prefix=THUMB

  esb
@ ARM:   esb   @ encoding: [0x10,0xf0,0x20,0xe3]
@ THUMB: esb.w @ encoding: [0xaf,0xf3,0x10,0x80]
