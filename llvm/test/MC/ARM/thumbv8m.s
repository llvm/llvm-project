// RUN: not llvm-mc -triple=thumbv8m.base -show-encoding < %s 2>%t \
// RUN:   | FileCheck --check-prefix=CHECK %s
// RUN:     FileCheck --check-prefix=UNDEF-BASELINE --check-prefix=UNDEF < %t %s
// RUN: not llvm-mc -triple=thumbv8m.main -show-encoding < %s 2>%t \
// RUN:   | FileCheck --check-prefix=CHECK %s
// RUN:     FileCheck --check-prefix=UNDEF-MAINLINE --check-prefix=UNDEF < %t %s

// Simple check that baseline is v6M and mainline is v7M
// UNDEF-BASELINE: error: instruction requires: thumb2
// UNDEF-MAINLINE-NOT: error: instruction requires:
mov.w r0, r0

// Check that .arm is invalid
// UNDEF: target does not support ARM mode
.arm

// Instruction availibility checks

// 'Barrier instructions'

// CHECK: isb	sy              @ encoding: [0xbf,0xf3,0x6f,0x8f]
isb sy

// 'Code optimization'

// CHECK: cbz r3, .Ltmp0      @ encoding: [0x03'A',0xb1'A']
// CHECK-NEXT:                @   fixup A - offset: 0, value: .Ltmp0, kind: fixup_arm_thumb_cb
cbz r3, 1f

// CHECK: cbnz r3, .Ltmp0     @ encoding: [0x03'A',0xb9'A']
// CHECK-NEXT:                @   fixup A - offset: 0, value: .Ltmp0, kind: fixup_arm_thumb_cb
cbnz r3, 1f

// CHECK: b.w .Ltmp0          @ encoding: [A,0xf0'A',A,0x90'A']
// CHECK-NEXT:                @   fixup A - offset: 0, value: .Ltmp0, kind: fixup_t2_uncondbranch
b.w 1f

// CHECK: sdiv r1, r2, r3     @ encoding: [0x92,0xfb,0xf3,0xf1]
sdiv r1, r2, r3

// CHECK: udiv r1, r2, r3     @ encoding: [0xb2,0xfb,0xf3,0xf1]
udiv r1, r2, r3

// 'Exclusives from ARMv7-M'

// CHECK: clrex               @ encoding: [0xbf,0xf3,0x2f,0x8f]
clrex

// CHECK: ldrex  r1, [r2, #4]        @ encoding: [0x52,0xe8,0x01,0x1f]
ldrex r1, [r2, #4]

// CHECK: ldrexb r1, [r2]            @ encoding: [0xd2,0xe8,0x4f,0x1f]
ldrexb r1, [r2]

// CHECK: ldrexh r1, [r2]            @ encoding: [0xd2,0xe8,0x5f,0x1f]
ldrexh r1, [r2]

// UNDEF-BASELINE: error: instruction requires: !armv*m thumb2
// UNDEF-MAINLINE: error: instruction requires: !armv*m
ldrexd r0, r1, [r2]

// CHECK: strex  r1, r2, [r3, #4]    @ encoding: [0x43,0xe8,0x01,0x21]
strex r1, r2, [r3, #4]

// CHECK: strexb r1, r2, [r3]        @ encoding: [0xc3,0xe8,0x41,0x2f]
strexb r1, r2, [r3]

// CHECK: strexh r1, r2, [r3]        @ encoding: [0xc3,0xe8,0x51,0x2f]
strexh r1, r2, [r3]

// UNDEF-BASELINE: error: instruction requires: !armv*m thumb2
// UNDEF-MAINLINE: error: instruction requires: !armv*m
strexd r0, r1, r2, [r3]

// 'XO generation'

// CHECK: movw r1, #65535            @ encoding: [0x4f,0xf6,0xff,0x71]
movw r1, #0xffff

// CHECK: movt r1, #65535            @ encoding: [0xcf,0xf6,0xff,0x71]
movt r1, #0xffff

// 'Acquire/Release from ARMv8-A'

// CHECK: lda r1, [r2]                @ encoding: [0xd2,0xe8,0xaf,0x1f]
lda r1, [r2]

// CHECK: ldab  r1, [r2]                @ encoding: [0xd2,0xe8,0x8f,0x1f]
ldab r1, [r2]

// CHECK: ldah  r1, [r2]                @ encoding: [0xd2,0xe8,0x9f,0x1f]
ldah r1, [r2]

// CHECK: stl r1, [r3]                @ encoding: [0xc3,0xe8,0xaf,0x1f]
stl r1, [r3]

// CHECK: stlb  r1, [r3]                @ encoding: [0xc3,0xe8,0x8f,0x1f]
stlb r1, [r3]

// CHECK: stlh  r1, [r3]                @ encoding: [0xc3,0xe8,0x9f,0x1f]
stlh r1, [r3]

// CHECK: ldaex r1, [r2]                @ encoding: [0xd2,0xe8,0xef,0x1f]
ldaex r1, [r2]

// CHECK: ldaexb  r1, [r2]                @ encoding: [0xd2,0xe8,0xcf,0x1f]
ldaexb r1, [r2]

// CHECK: ldaexh  r1, [r2]                @ encoding: [0xd2,0xe8,0xdf,0x1f]
ldaexh r1, [r2]

// UNDEF: error: instruction requires: !armv*m
ldaexd r0, r1, [r2]

// CHECK: stlex r1, r2, [r3]            @ encoding: [0xc3,0xe8,0xe1,0x2f]
stlex r1, r2, [r3]

// CHECK: stlexb  r1, r2, [r3]            @ encoding: [0xc3,0xe8,0xc1,0x2f]
stlexb r1, r2, [r3]

// CHECK: stlexh  r1, r2, [r3]            @ encoding: [0xc3,0xe8,0xd1,0x2f]
stlexh r1, r2, [r3]

// UNDEF: error: instruction requires: !armv*m
stlexd r0, r1, r2, [r2]
