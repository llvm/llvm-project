@ RUN:     llvm-mc -triple   armv8a-none-eabi -show-encoding %s      | FileCheck %s --check-prefix=ARM
@ RUN:     llvm-mc -triple thumbv8a-none-eabi -show-encoding %s      | FileCheck %s --check-prefix=THUMB
@ RUN: not llvm-mc -triple thumbv6m-none-eabi -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

csdb
ssbb
pssbb

@ ARM:   csdb   @ encoding: [0x14,0xf0,0x20,0xe3]
@ ARM:   ssbb   @ encoding: [0x40,0xf0,0x7f,0xf5]
@ ARM:   pssbb  @ encoding: [0x44,0xf0,0x7f,0xf5]

@ THUMB: csdb   @ encoding: [0xaf,0xf3,0x14,0x80]
@ THUMB: ssbb   @ encoding: [0xbf,0xf3,0x40,0x8f]
@ THUMB: pssbb  @ encoding: [0xbf,0xf3,0x44,0x8f]

@ ERROR:      error: instruction requires: thumb2
@ ERROR-NEXT: csdb
@ ERROR:      error: instruction requires: thumb2
@ ERROR-NEXT: ssbb
@ ERROR:      error: instruction requires: thumb2
@ ERROR-NEXT: pssbb
