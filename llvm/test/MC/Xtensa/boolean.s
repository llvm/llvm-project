# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+bool \
# RUN:     | FileCheck -check-prefixes=CHECK %s

.align	4
// CHECK: .p2align	4

LBL0:
// CHECK: LBL0:

all4 b1, b4
// CHECK: all4	b1, b4                          # encoding: [0x10,0x94,0x00]

all8 b1, b8
// CHECK: all8	b1, b8                          # encoding: [0x10,0xb8,0x00]

andb b1, b2, b3
// CHECK: andb	b1, b2, b3                      # encoding: [0x30,0x12,0x02]

andbc b1, b2, b3
// CHECK: andbc	b1, b2, b3                      # encoding: [0x30,0x12,0x12]

orb b1, b2, b3
// CHECK: orb	b1, b2, b3                      # encoding: [0x30,0x12,0x22]

orbc b1, b2, b3
// CHECK: orbc	b1, b2, b3                      # encoding: [0x30,0x12,0x32]

xorb b1, b2, b3
// CHECK: xorb	b1, b2, b3                      # encoding: [0x30,0x12,0x42]

any4 b1, b4
// CHECK: any4	b1, b4                          # encoding: [0x10,0x84,0x00]

any8 b1, b8
// CHECK: any8	b1, b8                          # encoding: [0x10,0xa8,0x00]

bt b1, LBL0
// CHECK: bt	b1, LBL0                        # encoding: [0x76,0x11,A]
// CHECK-NEXT: #   fixup A - offset: 0, value: LBL0, kind: fixup_xtensa_branch_8

bf b0, LBL0
// CHECK: bf	b0, LBL0                        # encoding: [0x76,0x00,A]
// CHECK-NEXT: #   fixup A - offset: 0, value: LBL0, kind: fixup_xtensa_branch_8

movf a2, a3, b1
// CHECK: movf	a2, a3, b1                      # encoding: [0x10,0x23,0xc3]

movt a3, a4, b2
// CHECK: movt	a3, a4, b2                      # encoding: [0x20,0x34,0xd3]

xsr a3, br
// CHECK: xsr	a3, br                          # encoding: [0x30,0x04,0x61]

xsr.br a3
// CHECK: xsr	a3, br                          # encoding: [0x30,0x04,0x61]

xsr a3, 4
// CHECK: xsr	a3, br                          # encoding: [0x30,0x04,0x61]
