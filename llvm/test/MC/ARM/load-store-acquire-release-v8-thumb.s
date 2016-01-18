@ RUN: llvm-mc -triple=thumbv8 -show-encoding < %s | FileCheck %s
@ RUN: not llvm-mc -triple=thumbv7 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V7
        ldaexb  r3, [r4]
        ldaexh  r2, [r5]
        ldaex  r1, [r7]
        ldaexd  r6, r7, [r8]

@ CHECK:  ldaexb	r3, [r4]                @ encoding: [0xd4,0xe8,0xcf,0x3f]
@ CHECK:  ldaexh	r2, [r5]                @ encoding: [0xd5,0xe8,0xdf,0x2f]
@ CHECK:  ldaex	r1, [r7]                @ encoding: [0xd7,0xe8,0xef,0x1f]
@ CHECK:  ldaexd	r6, r7, [r8]            @ encoding: [0xd8,0xe8,0xff,0x67]
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8

        stlexb  r1, r3, [r4]
        stlexh  r4, r2, [r5]
        stlex  r2, r1, [r7]
        stlexd  r6, r2, r3, [r8]
@ CHECK: stlexb r1, r3, [r4]            @ encoding: [0xc4,0xe8,0xc1,0x3f]
@ CHECK: stlexh r4, r2, [r5]            @ encoding: [0xc5,0xe8,0xd4,0x2f]
@ CHECK: stlex r2, r1, [r7]            @ encoding: [0xc7,0xe8,0xe2,0x1f]
@ CHECK: stlexd r6, r2, r3, [r8]        @ encoding: [0xc8,0xe8,0xf6,0x23]
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8

         lda r5, [r6]
         ldab r5, [r6]
         ldah r12, [r9]
@ CHECK: lda r5, [r6]                   @ encoding: [0xd6,0xe8,0xaf,0x5f]
@ CHECK: ldab r5, [r6]                  @ encoding: [0xd6,0xe8,0x8f,0x5f]
@ CHECK: ldah r12, [r9]                 @ encoding: [0xd9,0xe8,0x9f,0xcf]
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8

         stl r3, [r0]
         stlb r2, [r1]
         stlh r2, [r3]
@ CHECK: stl r3, [r0]                   @ encoding: [0xc0,0xe8,0xaf,0x3f]
@ CHECK: stlb r2, [r1]                  @ encoding: [0xc1,0xe8,0x8f,0x2f]
@ CHECK: stlh r2, [r3]                  @ encoding: [0xc3,0xe8,0x9f,0x2f]
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8
@ CHECK-V7: error: instruction requires: armv8
