# RUN: llvm-mc -triple thumbv7a -mattr=+dotprod --disassemble < %s | FileCheck %s
# RUN: llvm-mc -triple thumbv7a -mattr=-dotprod --disassemble < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

[0x21,0xfc,0x12,0x0d]
[0x21,0xfc,0x02,0x0d]
[0x22,0xfc,0x58,0x0d]
[0x22,0xfc,0x48,0x0d]
[0x21,0xfe,0x12,0x0d]
[0x21,0xfe,0x22,0x0d]
[0x22,0xfe,0x54,0x0d]
[0x22,0xfe,0x64,0x0d]

#CHECK: vudot.u8  d0, d1, d2
#CHECK: vsdot.s8  d0, d1, d2
#CHECK: vudot.u8  q0, q1, q4
#CHECK: vsdot.s8  q0, q1, q4
#CHECK: vudot.u8  d0, d1, d2[0]
#CHECK: vsdot.s8  d0, d1, d2[1]
#CHECK: vudot.u8  q0, q1, d4[0]
#CHECK: vsdot.s8  q0, q1, d4[1]

#CHECK-ERROR:  stc2  p13, c0, [r1], #-72
#CHECK-ERROR:  stc2  p13, c0, [r1], #-8
#CHECK-ERROR:  stc2  p13, c0, [r2], #-352
#CHECK-ERROR:  stc2  p13, c0, [r2], #-288
#CHECK-ERROR:  mcr2  p13, #1, r0, c1, c2, #0
#CHECK-ERROR:  cdp2  p13, #2, c0, c1, c2, #1
#CHECK-ERROR:  mcr2  p13, #1, r0, c2, c4, #2
#CHECK-ERROR:  cdp2  p13, #2, c0, c2, c4, #3
