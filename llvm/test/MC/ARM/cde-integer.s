// RUN: not llvm-mc -triple=thumbv8m.main -mattr=+cdecp0 -mattr=+cdecp1 -show-encoding < %s 2>%t | FileCheck %s
// RUN: FileCheck <%t --check-prefix=ERROR %s

// CHECK-LABEL: test_gcp
test_gcp:
// CHECK-NEXT: mrc p3, #1, r3, c15, c15, #5 @ encoding: [0x3f,0xee,0xbf,0x33]
mrc	p3, #1, r3, c15, c15, #5
// CHECK-NEXT: mcr2 p3, #2, r2, c7, c11, #7 @ encoding: [0x47,0xfe,0xfb,0x23]
mcr2 p3, #2, r2, c7, c11, #7

// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
mrc   p0, #1, r2, c3, c4, #5
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
ldc2  p1, c8, [r1, #4]
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
ldc2  p0, c7, [r2]
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
ldc2  p1, c6, [r3, #-224]
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
ldc2  p0, c5, [r4, #-120]!
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
ldc2l p1, c2, [r7, #4]
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
ldc2l p0, c1, [r8]
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
ldc2l p1, c0, [r9, #-224]
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
ldc2l p0, c1, [r10, #-120]!
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
stc2 p1, c8, [r1, #4]
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
stc2 p0, c7, [r2]
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
stc2 p1, c6, [r3, #-224]
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
stc2 p0, c5, [r4, #-120]!
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
stc2l p1, c2, [r7, #4]
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
stc2l p0, c1, [r8]
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
stc2l p1, c0, [r9, #-224]
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as GCP
stc2l p0, c1, [r10, #-120]!

// CHECK-LABEL: test_predication1:
test_predication1:
ittt eq
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: instructions in IT block must be predicable
cx1 p0, r3, #8191
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: instructions in IT block must be predicable
cx2 p0, r2, r3, #123
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: instructions in IT block must be predicable
cx3 p0, r1, r5, r7, #63
nop
nop
nop
ittt eq
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: instructions in IT block must be predicable
cx1d p0, r0, r1, #8191
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: instructions in IT block must be predicable
cx2d p0, r0, r1, r3, #123
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: instructions in IT block must be predicable
cx3d p0, r0, r1, r5, r7, #63
nop
nop
nop

// CHECK-LABEL: test_predication2:
test_predication2:
// CHECK: itte eq @ encoding: [0x06,0xbf]
itte eq
// CHECK-NEXT: cx1aeq p0, r3, #8191 @ encoding: [0x3f,0xfe,0xbf,0x30]
cx1aeq p0, r3, #8191
// CHECK-NEXT: cx2aeq p0, r2, r3, #123 @ encoding: [0x43,0xfe,0xbb,0x20]
cx2aeq p0, r2, r3, #123
// CHECK-NEXT: cx3ane p0, r1, r5, r7, #63 @ encoding: [0xf5,0xfe,0xb1,0x70]
cx3ane p0, r1, r5, r7, #63
// CHECK-NEXT: itte eq @ encoding: [0x06,0xbf]
itte eq
// CHECK-NEXT: cx1daeq p0, r0, r1, #8191 @ encoding: [0x3f,0xfe,0xff,0x00]
cx1daeq p0, r0, r1, #8191
// CHECK-NEXT: cx2daeq p0, r0, r1, r3, #123 @ encoding: [0x43,0xfe,0xfb,0x00]
cx2daeq p0, r0, r1, r3, #123
// CHECK-NEXT: cx3dane p0, r0, r1, r5, r7, #63 @ encoding: [0xf5,0xfe,0xf0,0x70]
cx3dane p0, r0, r1, r5, r7, #63


// CHECK-LABEL: test_cx1:
test_cx1:
// CHECK-NEXT:  cx1	p0, r3, #8191           @ encoding: [0x3f,0xee,0xbf,0x30]
cx1   p0, r3, #8191
// CHECK-NEXT: cx1a	p1, r2, #0              @ encoding: [0x00,0xfe,0x00,0x21]
cx1a  p1, r2, #0
// CHECK-NEXT: cx1d     p0, r4, r5, #1234       @ encoding: [0x09,0xee,0xd2,0x40]
cx1d  p0, r4, r5, #1234
// CHECK-NEXT: cx1da	p1, r2, r3, #1234       @ encoding: [0x09,0xfe,0xd2,0x21]
cx1da p1, r2, r3, #1234
// CHECK-NEXT: cx1	p0, apsr_nzcv, #8191    @ encoding: [0x3f,0xee,0xbf,0xf0]
cx1   p0, apsr_nzcv, #8191
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be in the range [p0, p7]
cx1   p8, r1, #1234
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: coprocessor must be configured as CDE
cx1   p2, r0, #1
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be an immediate in the range [0,8191]
cx1   p0, r1, #8192
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in the range [r0, r12], r14 or apsr_nzcv
cx1   p0, r13, #1234
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be a consecutive register
cx1d  p1, r0, #1234, #123
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be an even-numbered register in the range [r0, r10]
cx1d  p1, r1, #1234
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be a consecutive register
cx1d  p1, r2, r4, #1234
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be an even-numbered register in the range [r0, r10]
cx1da  p0, r1, #1234
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
cx1   p0, r0, r0, #1234
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
cx1d   p0, r0, r1, r2, #1234
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
cx1a   p0, r0, r2, #1234

// CHECK-LABEL: test_cx2:
test_cx2:
// CHECK-NEXT: cx2	p0, r3, r7, #0          @ encoding: [0x47,0xee,0x00,0x30]
cx2   p0, r3, r7, #0
// CHECK-NEXT: cx2a	p0, r1, r4, #511        @ encoding: [0x74,0xfe,0xbf,0x10]
cx2a  p0, r1, r4, #511
// CHECK-NEXT: cx2d	p0, r2, r3, r1, #123        @ encoding: [0x41,0xee,0xfb,0x20]
cx2d  p0, r2, r3, r1, #123
// CHECK-NEXT: cx2da p0, r2, r3, r7, #123        @ encoding: [0x47,0xfe,0xfb,0x20]
cx2da p0, r2, r3, r7, #123
// CHECK-NEXT: cx2da p1, r10, r11, apsr_nzcv, #123 @ encoding: [0x4f,0xfe,0xfb,0xa1]
cx2da p1, r10, r11, apsr_nzcv, #123

// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be an immediate in the range [0,511]
cx2   p0, r1, r4, #512
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be an even-numbered register in the range [r0, r10]
cx2d  p0, r12, r7, #123
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be an even-numbered register in the range [r0, r10]
cx2da  p0, r7, r7, #123
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be an even-numbered register in the range [r0, r10]
cx2da  p1, apsr_nzcv, r7, #123
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
cx2   p0, r0, r0, r7, #1
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be a consecutive register
cx2d   p0, r0, r0, r7, #1
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
cx2a   p0, r0, r2, r7, #1
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be a consecutive register
cx2da   p0, r0, r2, r7, #1

// CHECK-LABEL: test_cx3:
test_cx3:
// CHECK-NEXT: cx3 p0, r1, r2, r3, #0 @ encoding: [0x82,0xee,0x01,0x30]
cx3     p0, r1, r2, r3, #0
// CHECK-NEXT: cx3a p0, r1, r5, r7, #63 @ encoding: [0xf5,0xfe,0xb1,0x70]
cx3a    p0, r1, r5, r7, #63
// CHECK-NEXT: cx3d p1, r0, r1, r7, r1, #12 @ encoding: [0x97,0xee,0xc0,0x11]
cx3d    p1, r0, r1, r7, r1, #12
// CHECK-NEXT: cx3da p0, r8, r9, r2, r3, #12 @ encoding: [0x92,0xfe,0xc8,0x30]
cx3da   p0, r8, r9, r2, r3, #12
// CHECK-NEXT: cx3	p1, apsr_nzcv, r7, apsr_nzcv, #12 @ encoding: [0x97,0xee,0x8f,0xf1]
cx3     p1, apsr_nzcv, r7, apsr_nzcv, #12
// CHECK-NEXT: cx3d	p0, r8, r9, apsr_nzcv, apsr_nzcv, #12 @ encoding: [0x9f,0xee,0xc8,0xf0]
cx3d    p0, r8, r9, apsr_nzcv, apsr_nzcv, #12

// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be an immediate in the range [0,63]
cx3     p0, r1, r5, r7, #64
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be an even-numbered register in the range [r0, r10]
cx3da   p1, r14, r2, r3, #12
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be a register in the range [r0, r12], r14 or apsr_nzcv
cx3a    p0, r15, r2, r3, #12
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
cx2   p0, r0, r0, r7, r3, #1
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be a consecutive register
cx2d   p0, r0, r0, r7, r3, #1
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
cx3a    p0, r1, r2, r5, r7, #63
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: operand must be a consecutive register
cx3da   p0, r8, apsr_nzcv, r2, r3, #12

// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx1    p0, s0, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx1    p0, d0, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx1    p0, q0, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx1a   p0, s0, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx1a   p0, d0, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx1a   p0, q0, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx2    p0, s0, s1, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx2    p0, d0, d1, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx2    p0, q0, q1, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx2a   p0, s0, s1, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx2a   p0, d0, d1, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx2    p0, q0, q1, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx3    p0, s0, s1, s2, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx3    p0, d0, d1, d2, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx3    p0, q0, q1, q2, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx3a   p0, s0, s1, s2, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx3a   p0, d0, d1, d2, #0
// ERROR: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
vcx3a   p0, q0, q1, q2, #0
