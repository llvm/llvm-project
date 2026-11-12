@ PR18921
@ RUN: llvm-mc -triple=armv7-linux-gnueabi -show-encoding < %s | FileCheck %s
.text

@ CHECK-NOT: .code	16


@ CHECK: ldrd	r0, r1, [r10, #32]!     @ encoding: [0xd0,0x02,0xea,0xe1]
@ CHECK: ldrd	r0, r1, [r10], #32      @ encoding: [0xd0,0x02,0xca,0xe0]
@ CHECK: ldrd	r0, r1, [r10, #32]      @ encoding: [0xd0,0x02,0xca,0xe1]
        ldrd    r0, [r10, #32]!
        ldrd    r0, [r10], #32
        ldrd    r0, [r10, #32]

@ CHECK: strd	r0, r1, [r10, #32]!     @ encoding: [0xf0,0x02,0xea,0xe1]
@ CHECK: strd	r0, r1, [r10], #32      @ encoding: [0xf0,0x02,0xca,0xe0]
@ CHECK: strd	r0, r1, [r10, #32]      @ encoding: [0xf0,0x02,0xca,0xe1]
        strd    r0, [r10, #32]!
        strd    r0, [r10], #32
        strd    r0, [r10, #32]
