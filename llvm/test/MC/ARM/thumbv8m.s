// RUN: not llvm-mc -triple=thumbv8m.base -show-encoding < %s 2>%t \
// RUN:   | FileCheck --check-prefix=CHECK-BASELINE --check-prefix=CHECK %s
// RUN:     FileCheck --check-prefix=UNDEF-BASELINE --check-prefix=UNDEF < %t %s
// RUN: not llvm-mc -triple=thumbv8m.main -show-encoding < %s 2>%t \
// RUN:   | FileCheck --check-prefix=CHECK-MAINLINE --check-prefix=CHECK %s
// RUN:     FileCheck --check-prefix=UNDEF-MAINLINE --check-prefix=UNDEF < %t %s
// RUN: not llvm-mc -triple=thumbv8m.main -mattr=+dsp,+t2xtpk -show-encoding < %s 2>%t \
// RUN:   | FileCheck --check-prefix=CHECK-MAINLINE_DSP --check-prefix=CHECK %s
// RUN:     FileCheck --check-prefix=UNDEF-MAINLINE_DSP --check-prefix=UNDEF < %t %s

// Simple check that baseline is v6M and mainline is v7M
// UNDEF-BASELINE: error: instruction requires: thumb2
// UNDEF-MAINLINE-NOT: error: instruction requires:
// UNDEF-MAINLINE_DSP-NOT: error: instruction requires:
mov.w r0, r0

// Check that .arm is invalid
// UNDEF: target does not support ARM mode
.arm

// And only +dsp,+t2xtpk has DSP and t2xtpk instructions
// UNDEF-BASELINE: error: instruction requires: arm-mode
// UNDEF-MAINLINE: error: instruction requires: arm-mode
// UNDEF-MAINLINE_DSP-NOT: error: instruction requires:
qadd16 r0, r0, r0
// UNDEF-BASELINE: error: instruction requires: arm-mode
// UNDEF-MAINLINE: error: instruction requires: arm-mode
// UNDEF-MAINLINE_DSP-NOT: error: instruction requires:
uxtab16 r0, r1, r2

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

// ARMv8-M Security Extensions

// CHECK: sg                         @ encoding: [0x7f,0xe9,0x7f,0xe9]
sg

// CHECK: bxns r0                    @ encoding: [0x04,0x47]
bxns r0

// UNDEF-BASELINE: error: invalid operand for instruction
// UNDEF-BASELINE: error: conditional execution not supported in Thumb1
// CHECK-MAINLINE: it eq                      @ encoding: [0x08,0xbf]
// CHECK-MAINLINE: bxnseq r1                  @ encoding: [0x0c,0x47]
it eq
bxnseq r1

// CHECK: bxns lr                    @ encoding: [0x74,0x47]
bxns lr

// CHECK: blxns r0                   @ encoding: [0x84,0x47]
blxns r0

// UNDEF-BASELINE: error: invalid operand for instruction
// UNDEF-BASELINE: error: conditional execution not supported in Thumb1
// CHECK-MAINLINE: it eq                      @ encoding: [0x08,0xbf]
// CHECK-MAINLINE: blxnseq r1                 @ encoding: [0x8c,0x47]
it eq
blxnseq r1

// CHECK: tt r0, r1                  @ encoding: [0x41,0xe8,0x00,0xf0]
tt r0, r1

// CHECK: tt r0, sp                  @ encoding: [0x4d,0xe8,0x00,0xf0]
tt r0, sp

// CHECK: tta r0, r1                 @ encoding: [0x41,0xe8,0x80,0xf0]
tta r0, r1

// CHECK: ttt r0, r1                 @ encoding: [0x41,0xe8,0x40,0xf0]
ttt r0, r1

// CHECK: ttat r0, r1                @ encoding: [0x41,0xe8,0xc0,0xf0]
ttat r0, r1

// 'Lazy Load/Store Multiple'

// UNDEF-BASELINE: error: instruction requires: armv8m.main
// CHECK-MAINLINE: vlldm r5          @ encoding: [0x35,0xec,0x00,0x0a]
// CHECK-MAINLINE_DSP: vlldm r5      @ encoding: [0x35,0xec,0x00,0x0a]
vlldm r5

// UNDEF-BASELINE: error: instruction requires: armv8m.main
// CHECK-MAINLINE: vlstm r10         @ encoding: [0x2a,0xec,0x00,0x0a]
// CHECK-MAINLINE_DSP: vlstm r10     @ encoding: [0x2a,0xec,0x00,0x0a]
vlstm r10

// New SYSm's

MRS r1, MSP_NS
// CHECK: mrs r1, msp_ns             @ encoding: [0xef,0xf3,0x88,0x81]
MSR PSP_NS, r2
// CHECK: msr psp_ns, r2             @ encoding: [0x82,0xf3,0x89,0x88]
MRS r3, PRIMASK_NS
// CHECK: mrs r3, primask_ns         @ encoding: [0xef,0xf3,0x90,0x83]
MSR CONTROL_NS, r4
// CHECK: msr control_ns, r4         @ encoding: [0x84,0xf3,0x94,0x88]
MRS r5, SP_NS
// CHECK: mrs r5, sp_ns              @ encoding: [0xef,0xf3,0x98,0x85]
MRS r6,MSPLIM
// CHECK: mrs r6, msplim             @ encoding: [0xef,0xf3,0x0a,0x86]
MRS r7,PSPLIM
// CHECK: mrs r7, psplim             @ encoding: [0xef,0xf3,0x0b,0x87]
MSR MSPLIM,r8
// CHECK: msr msplim, r8             @ encoding: [0x88,0xf3,0x0a,0x88]
MSR PSPLIM,r9
// CHECK: msr psplim, r9             @ encoding: [0x89,0xf3,0x0b,0x88]

MRS r10, MSPLIM_NS
// CHECK-MAINLINE: mrs r10, msplim_ns    @ encoding: [0xef,0xf3,0x8a,0x8a]
// UNDEF-BASELINE: error: invalid operand for instruction
MSR PSPLIM_NS, r11
// CHECK-MAINLINE: msr psplim_ns, r11    @ encoding: [0x8b,0xf3,0x8b,0x88]
// UNDEF-BASELINE: error: invalid operand for instruction
MRS r12, BASEPRI_NS
// CHECK-MAINLINE: mrs r12, basepri_ns   @ encoding: [0xef,0xf3,0x91,0x8c]
// UNDEF-BASELINE: error: invalid operand for instruction
MRS r12, BASEPRI_MAX_NS
// CHECK-MAINLINE: mrs r12, basepri_max_ns @ encoding: [0xef,0xf3,0x92,0x8c]
// UNDEF-BASELINE: error: invalid operand for instruction
MSR FAULTMASK_NS, r14
// CHECK-MAINLINE: msr faultmask_ns, lr  @ encoding: [0x8e,0xf3,0x93,0x88]
// UNDEF-BASELINE: error: invalid operand for instruction

// Invalid operand tests
// UNDEF: error: invalid operand for instruction
// UNDEF:     sg #0
sg #0
// UNDEF: error: invalid operand for instruction
// UNDEF:     sg r0
sg r0
// UNDEF: error: invalid operand for instruction
// UNDEF:     bxns r0, r1
bxns r0, r1
// UNDEF: error: invalid operand for instruction
// UNDEF:     blxns r0, #0
blxns r0, #0
// UNDEF: error: invalid operand for instruction
// UNDEF:     blxns label
blxns label
// UNDEF: error: invalid operand for instruction
// UNDEF:     tt r0, r1, r2
tt r0, r1, r2
// UNDEF: error: invalid operand for instruction
// UNDEF:     tt r0, [r1]
tt r0, [r1]
// UNDEF: error: invalid operand for instruction
// UNDEF:     tt r0, r1, #4
tt r0, r1, #4
// UNDEF: error: invalid operand for instruction
// UNDEF:     tt r0, #4
tt r0, #4

// Unpredictable operands
// UNDEF: error: invalid operand for instruction
// UNDEF:     blxns pc
blxns pc
// UNDEF: error: invalid operand for instruction
// UNDEF:     tt sp, r0
tt sp, r0
// UNDEF: error: invalid operand for instruction
// UNDEF:     tt pc, r0
tt pc, r0
// UNDEF: error: invalid operand for instruction
// UNDEF:     tt r0, pc
tt r0, pc

// UNDEF: error: invalid operand for instruction
// UNDEF:     vlldm pc
vlldm pc

// UNDEF: error: invalid operand for instruction
// UNDEF:     vlstm pc
vlstm pc
