@ RUN: llvm-mc -triple=armv7-linux-gnueabi -show-encoding < %s | FileCheck %s

@ CHECK: ldc	p12, c4, [r0, #4]       @ encoding: [0x01,0x4c,0x90,0xed]
@ CHECK: stc	p14, c6, [r2, #-224]    @ encoding: [0x38,0x6e,0x02,0xed]

        ldc p12, cr4, [r0, #4]
        stc p14, cr6, [r2, #-224]
@ RUN: llvm-mc -triple=armv7-linux-gnueabi -show-encoding < %s | FileCheck %s

@ CHECK: ldc	p12, c4, [r0, #4]       @ encoding: [0x01,0x4c,0x90,0xed]
@ CHECK: stc	p14, c6, [r2, #-224]    @ encoding: [0x38,0x6e,0x02,0xed]

        ldc p12, cr4, [r0, #4]
        stc p14, cr6, [r2, #-224]
