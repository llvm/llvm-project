@ RUN: llvm-mc -triple=thumbv8 -show-encoding < %s | FileCheck %s
@ RUN: not llvm-mc -triple=thumbv7 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V7
@ RUN: not llvm-mc -triple=thumbv8 -mattr=-crc -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOCRC
        crc32b  r0, r1, r2
        crc32h  r0, r1, r2
        crc32w  r0, r1, r2

@ CHECK:  crc32b    r0, r1, r2              @ encoding: [0xc1,0xfa,0x82,0xf0]
@ CHECK:  crc32h    r0, r1, r2              @ encoding: [0xc1,0xfa,0x92,0xf0]
@ CHECK:  crc32w    r0, r1, r2              @ encoding: [0xc1,0xfa,0xa2,0xf0]
@ CHECK-V7: error: instruction requires: crc
@ CHECK-V7: error: instruction requires: crc
@ CHECK-V7: error: instruction requires: crc
@ CHECK-NOCRC: error: instruction requires: crc
@ CHECK-NOCRC: error: instruction requires: crc
@ CHECK-NOCRC: error: instruction requires: crc

        crc32cb  r0, r1, r2
        crc32ch  r0, r1, r2
        crc32cw  r0, r1, r2

@ CHECK:  crc32cb   r0, r1, r2              @ encoding: [0xd1,0xfa,0x82,0xf0]
@ CHECK:  crc32ch   r0, r1, r2              @ encoding: [0xd1,0xfa,0x92,0xf0]
@ CHECK:  crc32cw   r0, r1, r2              @ encoding: [0xd1,0xfa,0xa2,0xf0]
@ CHECK-V7: error: instruction requires: crc
@ CHECK-V7: error: instruction requires: crc
@ CHECK-V7: error: instruction requires: crc
@ CHECK-NOCRC: error: instruction requires: crc
@ CHECK-NOCRC: error: instruction requires: crc
@ CHECK-NOCRC: error: instruction requires: crc
