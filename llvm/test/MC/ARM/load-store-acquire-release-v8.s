@ RUN: not llvm-mc -triple=armv8 -show-encoding < %s 2> %t | FileCheck %s
@ RUN: FileCheck %s < %t --check-prefix=CHECK-ERROR
@ RUN: not llvm-mc -triple=armv7 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V7
        ldaexb  r3, [r4]
        ldaexh  r2, [r5]
        ldaex  r1, [r7]
        ldaexd  r6, r7, [r8]

@ CHECK: ldaexb r3, [r4]                @ encoding: [0x9f,0x3e,0xd4,0xe1]
@ CHECK: ldaexh r2, [r5]                @ encoding: [0x9f,0x2e,0xf5,0xe1]
@ CHECK: ldaex r1, [r7]                @ encoding: [0x9f,0x1e,0x97,0xe1]
@ CHECK: ldaexd r6, r7, [r8]            @ encoding: [0x9f,0x6e,0xb8,0xe1]
@ CHECK-V7: instruction requires: acquire/release
@ CHECK-V7: instruction requires: acquire/release
@ CHECK-V7: instruction requires: acquire/release
@ CHECK-V7: instruction requires: acquire/release

        ldaexd  r2, r4, [r8]
@ CHECK-ERROR: error: destination operands must be sequential

@ GNU alias
        ldaexd  r6, [r8]
@ CHECK:  ldaexd	r6, r7, [r8]            @ encoding: [0x9f,0x6e,0xb8,0xe1]

        stlexb  r1, r3, [r4]
        stlexh  r4, r2, [r5]
        stlex  r2, r1, [r7]
        stlexd  r6, r2, r3, [r8]
@ CHECK: stlexb r1, r3, [r4]            @ encoding: [0x93,0x1e,0xc4,0xe1]
@ CHECK: stlexh r4, r2, [r5]            @ encoding: [0x92,0x4e,0xe5,0xe1]
@ CHECK: stlex r2, r1, [r7]            @ encoding: [0x91,0x2e,0x87,0xe1]
@ CHECK: stlexd r6, r2, r3, [r8]        @ encoding: [0x92,0x6e,0xa8,0xe1]
@ CHECK-V7: instruction requires: acquire/release
@ CHECK-V7: instruction requires: acquire/release
@ CHECK-V7: instruction requires: acquire/release
@ CHECK-V7: instruction requires: acquire/release

        stlexd  r6, r2, r4, [r8]
@ CHECK-ERROR: error: source operands must be sequential

@ GNU alias
        stlexd  r6, r2, [r8]
@ CHECK: stlexd r6, r2, r3, [r8]        @ encoding: [0x92,0x6e,0xa8,0xe1]

         lda r5, [r6]
         ldab r5, [r6]
         ldah r12, [r9]
@ CHECK: lda r5, [r6]                   @ encoding: [0x9f,0x5c,0x96,0xe1]
@ CHECK: ldab r5, [r6]                  @ encoding: [0x9f,0x5c,0xd6,0xe1]
@ CHECK: ldah r12, [r9]                 @ encoding: [0x9f,0xcc,0xf9,0xe1]
@ CHECK-V7: instruction requires: acquire/release
@ CHECK-V7: instruction requires: acquire/release
@ CHECK-V7: instruction requires: acquire/release

         stl r3, [r0]
         stlb r2, [r1]
         stlh r2, [r3]
@ CHECK: stl r3, [r0]                   @ encoding: [0x93,0xfc,0x80,0xe1]
@ CHECK: stlb r2, [r1]                  @ encoding: [0x92,0xfc,0xc1,0xe1]
@ CHECK: stlh r2, [r3]                  @ encoding: [0x92,0xfc,0xe3,0xe1]
@ CHECK-V7: instruction requires: acquire/release
@ CHECK-V7: instruction requires: acquire/release
@ CHECK-V7: instruction requires: acquire/release
