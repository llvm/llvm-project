@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding %s | FileCheck %s

@ Post-indexed
@ CHECK: ldrt  r1, [r0], r2 @ encoding: [0x02,0x10,0xb0,0xe6]
@ CHECK: ldrt  r1, [r0], r2, lsr #3 @ encoding: [0xa2,0x11,0xb0,0xe6]
@ CHECK: ldrt  r1, [r0], #4 @ encoding: [0x04,0x10,0xb0,0xe4]
@ CHECK: ldrt  r1, [r0], #0 @ encoding: [0x00,0x10,0xb0,0xe4]
@ CHECK: ldrbt  r1, [r0], r2 @ encoding: [0x02,0x10,0xf0,0xe6]
@ CHECK: ldrbt  r1, [r0], r2, lsr #3 @ encoding: [0xa2,0x11,0xf0,0xe6]
@ CHECK: ldrbt  r1, [r0], #4 @ encoding: [0x04,0x10,0xf0,0xe4]
@ CHECK: ldrbt  r1, [r0], #0 @ encoding: [0x00,0x10,0xf0,0xe4]
@ CHECK: strt  r1, [r0], r2 @ encoding: [0x02,0x10,0xa0,0xe6]
@ CHECK: strt  r1, [r0], r2, lsr #3 @ encoding: [0xa2,0x11,0xa0,0xe6]
@ CHECK: strt  r1, [r0], #4 @ encoding: [0x04,0x10,0xa0,0xe4]
@ CHECK: strt  r1, [r0], #0 @ encoding: [0x00,0x10,0xa0,0xe4]
@ CHECK: strbt  r1, [r0], r2 @ encoding: [0x02,0x10,0xe0,0xe6]
@ CHECK: strbt  r1, [r0], r2, lsr #3 @ encoding: [0xa2,0x11,0xe0,0xe6]
@ CHECK: strbt  r1, [r0], #4 @ encoding: [0x04,0x10,0xe0,0xe4]
@ CHECK: strbt  r1, [r0], #0 @ encoding: [0x00,0x10,0xe0,0xe4]
        ldrt  r1, [r0], r2
        ldrt  r1, [r0], r2, lsr #3
        ldrt  r1, [r0], #4
        ldrt  r1, [r0]
        ldrbt  r1, [r0], r2
        ldrbt  r1, [r0], r2, lsr #3
        ldrbt  r1, [r0], #4
        ldrbt  r1, [r0]
        strt  r1, [r0], r2
        strt  r1, [r0], r2, lsr #3
        strt  r1, [r0], #4
        strt  r1, [r0]
        strbt  r1, [r0], r2
        strbt  r1, [r0], r2, lsr #3
        strbt  r1, [r0], #4
        strbt  r1, [r0]

@ Pre-indexed
@ CHECK: ldr  r1, [r0, r2, lsr #3]! @ encoding: [0xa2,0x11,0xb0,0xe7]
@ CHECK: ldrb  r1, [r0, r2, lsr #3]! @ encoding: [0xa2,0x11,0xf0,0xe7]
        ldr  r1, [r0, r2, lsr #3]!
        ldrb  r1, [r0, r2, lsr #3]!

