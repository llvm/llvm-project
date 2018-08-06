@ RUN:     llvm-mc -triple   armv8a-none-eabi -show-encoding %s      | FileCheck %s --check-prefix=ARM
@ RUN:     llvm-mc -triple thumbv8a-none-eabi -show-encoding %s      | FileCheck %s --check-prefix=THUMB
@ RUN: not llvm-mc -triple thumbv6m-none-eabi -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

  csdb
@ ARM:   csdb   @ encoding: [0x14,0xf0,0x20,0xe3]
@ THUMB: csdb   @ encoding: [0xaf,0xf3,0x14,0x80]
@ ERROR: error: instruction requires: thumb2
