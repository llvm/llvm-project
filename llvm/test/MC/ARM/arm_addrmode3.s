@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding %s | FileCheck %s

@ CHECK: ldrsbt  r1, [r0], r2 @ encoding: [0xd2,0x10,0xb0,0xe0]
@ CHECK: ldrsbt  r1, [r0], #4 @ encoding: [0xd4,0x10,0xf0,0xe0]
@ CHECK: ldrsht  r1, [r0], r2 @ encoding: [0xf2,0x10,0xb0,0xe0]
@ CHECK: ldrsht  r1, [r0], #4 @ encoding: [0xf4,0x10,0xf0,0xe0]
@ CHECK: ldrht  r1, [r0], r2 @ encoding: [0xb2,0x10,0xb0,0xe0]
@ CHECK: ldrht  r1, [r0], #4 @ encoding: [0xb4,0x10,0xf0,0xe0]
@ CHECK: strht  r1, [r0], r2 @ encoding: [0xb2,0x10,0xa0,0xe0]
@ CHECK: strht  r1, [r0], #4 @ encoding: [0xb4,0x10,0xe0,0xe0]
        ldrsbt  r1, [r0], r2
        ldrsbt  r1, [r0], #4
        ldrsht  r1, [r0], r2
        ldrsht  r1, [r0], #4
        ldrht  r1, [r0], r2
        ldrht  r1, [r0], #4
        strht  r1, [r0], r2
        strht  r1, [r0], #4
