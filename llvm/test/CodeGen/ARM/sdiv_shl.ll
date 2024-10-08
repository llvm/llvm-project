; RUN: llc -mtriple armv7-linux -mattr=+neon %s -o - | FileCheck %s --check-prefix=LE
; RUN: llc -mtriple armebv7-linux -mattr=+neon %s -o - | FileCheck %s --check-prefix=BE

; The key is the last vrev64 should be vrev64.16 instead of vrev64.32

define void @sdiv_shl(ptr %x, ptr %y) nounwind {
; LE-LABEL: sdiv_shl:
; LE:       @ %bb.0: @ %entry
; LE-NEXT:    adr r2, .LCPI0_0
; LE-NEXT:    vld1.64 {d18, d19}, [r1]
; LE-NEXT:    adr r1, .LCPI0_1
; LE-NEXT:    vld1.64 {d16, d17}, [r2:128]
; LE-NEXT:    vshr.s16 q10, q9, #15
; LE-NEXT:    vneg.s16 q8, q8
; LE-NEXT:    vld1.64 {d22, d23}, [r1:128]
; LE-NEXT:    adr r1, .LCPI0_2
; LE-NEXT:    vshl.u16 q8, q10, q8
; LE-NEXT:    vneg.s16 q10, q11
; LE-NEXT:    vadd.i16 q8, q9, q8
; LE-NEXT:    vshl.s16 q8, q8, q10
; LE-NEXT:    vld1.64 {d20, d21}, [r1:128]
; LE-NEXT:    vbit q8, q9, q10
; LE-NEXT:    vst1.64 {d16, d17}, [r0]
; LE:         .LCPI0_0:
; LE-NEXT:    .short 16 @ 0x10
; LE-NEXT:    .short 14 @ 0xe
; LE-NEXT:    .short 15 @ 0xf
; LE-NEXT:    .short 13 @ 0xd
; LE-NEXT:    .short 12 @ 0xc
; LE-NEXT:    .short 10 @ 0xa
; LE-NEXT:    .short 11 @ 0xb
; LE-NEXT:    .short 9 @ 0x9
; LE-NEXT:  .LCPI0_1:
; LE-NEXT:    .short 0 @ 0x0
; LE-NEXT:    .short 2 @ 0x2
; LE-NEXT:    .short 1 @ 0x1
; LE-NEXT:    .short 3 @ 0x3
; LE-NEXT:    .short 4 @ 0x4
; LE-NEXT:    .short 6 @ 0x6
; LE-NEXT:    .short 5 @ 0x5
; LE-NEXT:    .short 7 @ 0x7
; LE-NEXT:  .LCPI0_2:
; LE-NEXT:    .short 65535 @ 0xffff
; LE-NEXT:    .short 0 @ 0x0
; LE-NEXT:    .short 0 @ 0x0
; LE-NEXT:    .short 0 @ 0x0
; LE-NEXT:    .short 0 @ 0x0
; LE-NEXT:    .short 0 @ 0x0
; LE-NEXT:    .short 0 @ 0x0
; LE-NEXT:    .short 0 @ 0x0
;
; BE-LABEL: sdiv_shl:
; BE:       @ %bb.0: @ %entry
; BE-NEXT:    adr r2, .LCPI0_0
; BE-NEXT:    vld1.64 {d18, d19}, [r1]
; BE-NEXT:    adr r1, .LCPI0_1
; BE-NEXT:    vld1.64 {d16, d17}, [r2:128]
; BE-NEXT:    vrev64.16 q8, q8
; BE-NEXT:    vrev64.16 q9, q9
; BE-NEXT:    vneg.s16 q8, q8
; BE-NEXT:    vld1.64 {d20, d21}, [r1:128]
; BE-NEXT:    adr r1, .LCPI0_2
; BE-NEXT:    vshr.s16 q11, q9, #15
; BE-NEXT:    vrev64.16 q10, q10
; BE-NEXT:    vshl.u16 q8, q11, q8
; BE-NEXT:    vld1.64 {d22, d23}, [r1:128]
; BE-NEXT:    vneg.s16 q10, q10
; BE-NEXT:    vrev64.16 q11, q11
; BE-NEXT:    vadd.i16 q8, q9, q8
; BE-NEXT:    vshl.s16 q8, q8, q10
; BE-NEXT:    vbit q8, q9, q11
; BE-NEXT:    vrev64.16 q8, q8
; BE-NEXT:    vst1.64 {d16, d17}, [r0]
; BE:         .LCPI0_0:
; BE-NEXT:    .short 16 @ 0x10
; BE-NEXT:    .short 14 @ 0xe
; BE-NEXT:    .short 15 @ 0xf
; BE-NEXT:    .short 13 @ 0xd
; BE-NEXT:    .short 12 @ 0xc
; BE-NEXT:    .short 10 @ 0xa
; BE-NEXT:    .short 11 @ 0xb
; BE-NEXT:    .short 9 @ 0x9
; BE-NEXT:  .LCPI0_1:
; BE-NEXT:    .short 0 @ 0x0
; BE-NEXT:    .short 2 @ 0x2
; BE-NEXT:    .short 1 @ 0x1
; BE-NEXT:    .short 3 @ 0x3
; BE-NEXT:    .short 4 @ 0x4
; BE-NEXT:    .short 6 @ 0x6
; BE-NEXT:    .short 5 @ 0x5
; BE-NEXT:    .short 7 @ 0x7
; BE-NEXT:  .LCPI0_2:
; BE-NEXT:    .short 65535 @ 0xffff
; BE-NEXT:    .short 0 @ 0x0
; BE-NEXT:    .short 0 @ 0x0
; BE-NEXT:    .short 0 @ 0x0
; BE-NEXT:    .short 0 @ 0x0
; BE-NEXT:    .short 0 @ 0x0
; BE-NEXT:    .short 0 @ 0x0
; BE-NEXT:    .short 0 @ 0x0
entry:
  %0 = load <8 x i16>, ptr %y, align 8
  %div = sdiv <8 x i16> %0, <i16 1, i16 4, i16 2, i16 8, i16 16, i16 64, i16 32, i16 128>
  store <8 x i16> %div, ptr %x, align 8
  ret void
}
