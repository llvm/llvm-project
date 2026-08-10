; Test that DAGCombiner gets helped by computeKnownBitsForTargetNode() with
; vector intrinsics.
;
; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 < %s  | FileCheck %s

declare <8 x i16> @llvm.s390.vuphb(<16 x i8>)
declare <8 x i16> @llvm.s390.vuplhb(<16 x i8>)

; VUPHB (used operand elements are 0)
define <8 x i16> @f0() {
; CHECK-LABEL: f0:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <8 x i16> @llvm.s390.vuphb(<16 x i8>
                                         <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
                                          i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  %and = and <8 x i16> %unp, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; VUPHB (used operand elements are 1)
; NOTE: The AND is optimized away, but instead of replicating '1' into <8 x
; i16>, the original vector constant is put in the constant pool and then
; unpacked (repeated in more test cases below).
define <8 x i16> @f1() {
; CHECK-LABEL: f1:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  larl %r1, .LCPI
; CHECK-NEXT:  vl %v0, 0(%r1)
; CHECK-NEXT:  vuphb %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <8 x i16> @llvm.s390.vuphb(<16 x i8>
                                         <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                                          i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>)
  %and = and <8 x i16> %unp, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; VUPLHB (used operand elements are 0)
define <8 x i16> @f2() {
; CHECK-LABEL: f2:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <8 x i16> @llvm.s390.vuplhb(<16 x i8>
                                          <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
                                           i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  %and = and <8 x i16> %unp, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; VUPLHB (used operand elements are 1)
define <8 x i16> @f3() {
; CHECK-LABEL: f3:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  larl %r1, .LCPI
; CHECK-NEXT:  vl %v0, 0(%r1)
; CHECK-NEXT:  vuplhb %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <8 x i16> @llvm.s390.vuplhb(<16 x i8>
                                          <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                                           i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>)
  %and = and <8 x i16> %unp, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

declare <4 x i32> @llvm.s390.vuphh(<8 x i16>)
declare <4 x i32> @llvm.s390.vuplhh(<8 x i16>)

; VUPHH (used operand elements are 0)
define <4 x i32> @f4() {
; CHECK-LABEL: f4:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <4 x i32> @llvm.s390.vuphh(<8 x i16>
                                         <i16 0, i16 0, i16 0, i16 0,
                                          i16 1, i16 1, i16 1, i16 1>)
  %and = and <4 x i32> %unp, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; VUPHH (used operand elements are 1)
define <4 x i32> @f5() {
; CHECK-LABEL: f5:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  larl %r1, .LCPI
; CHECK-NEXT:  vl %v0, 0(%r1)
; CHECK-NEXT:  vuphh %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <4 x i32> @llvm.s390.vuphh(<8 x i16>
                                         <i16 1, i16 1, i16 1, i16 1,
                                          i16 0, i16 0, i16 0, i16 0>)
  %and = and <4 x i32> %unp, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; VUPLHH (used operand elements are 0)
define <4 x i32> @f6() {
; CHECK-LABEL: f6:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <4 x i32> @llvm.s390.vuplhh(<8 x i16>
                                          <i16 0, i16 0, i16 0, i16 0,
                                           i16 1, i16 1, i16 1, i16 1>)
  %and = and <4 x i32> %unp, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; VUPLHH (used operand elements are 1)
define <4 x i32> @f7() {
; CHECK-LABEL: f7:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  larl %r1, .LCPI
; CHECK-NEXT:  vl %v0, 0(%r1)
; CHECK-NEXT:  vuplhh %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <4 x i32> @llvm.s390.vuplhh(<8 x i16>
                                          <i16 1, i16 1, i16 1, i16 1,
                                           i16 0, i16 0, i16 0, i16 0>)
  %and = and <4 x i32> %unp, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

declare <2 x i64> @llvm.s390.vuphf(<4 x i32>)
declare <2 x i64> @llvm.s390.vuplhf(<4 x i32>)

; VUPHF (used operand elements are 0)
define <2 x i64> @f8() {
; CHECK-LABEL: f8:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <2 x i64> @llvm.s390.vuphf(<4 x i32> <i32 0, i32 0, i32 1, i32 1>)
  %and = and <2 x i64> %unp, <i64 1, i64 1>
  ret <2 x i64> %and
}

; VUPHF (used operand elements are 1)
define <2 x i64> @f9() {
; CHECK-LABEL: f9:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  larl %r1, .LCPI
; CHECK-NEXT:  vl %v0, 0(%r1)
; CHECK-NEXT:  vuphf %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <2 x i64> @llvm.s390.vuphf(<4 x i32> <i32 1, i32 1, i32 0, i32 0>)
  %and = and <2 x i64> %unp, <i64 1, i64 1>
  ret <2 x i64> %and
}

; VUPLHF (used operand elements are 0)
define <2 x i64> @f10() {
; CHECK-LABEL: f10:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <2 x i64> @llvm.s390.vuplhf(<4 x i32> <i32 0, i32 0, i32 1, i32 1>)
  %and = and <2 x i64> %unp, <i64 1, i64 1>
  ret <2 x i64> %and
}

; VUPLHF (used operand elements are 1)
define <2 x i64> @f11() {
; CHECK-LABEL: f11:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  larl %r1, .LCPI
; CHECK-NEXT:  vl %v0, 0(%r1)
; CHECK-NEXT:  vuplhf %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <2 x i64> @llvm.s390.vuplhf(<4 x i32> <i32 1, i32 1, i32 0, i32 0>)
  %and = and <2 x i64> %unp, <i64 1, i64 1>
  ret <2 x i64> %and
}

declare <8 x i16> @llvm.s390.vuplb(<16 x i8>)
declare <8 x i16> @llvm.s390.vupllb(<16 x i8>)

; VUPLB (used operand elements are 0)
define <8 x i16> @f12() {
; CHECK-LABEL: f12:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <8 x i16> @llvm.s390.vuplb(<16 x i8>
                                         <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                                          i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>)

  %and = and <8 x i16> %unp, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; VUPLB (used operand elements are 1)
define <8 x i16> @f13() {
; CHECK-LABEL: f13:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  larl %r1, .LCPI
; CHECK-NEXT:  vl %v0, 0(%r1)
; CHECK-NEXT:  vuplb %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <8 x i16> @llvm.s390.vuplb(<16 x i8>
                                         <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
                                          i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  %and = and <8 x i16> %unp, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; VUPLLB (used operand elements are 0)
define <8 x i16> @f14() {
; CHECK-LABEL: f14:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <8 x i16> @llvm.s390.vupllb(<16 x i8>
                                         <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                                          i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>)
  %and = and <8 x i16> %unp, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

; VUPLLB (used operand elements are 1)
define <8 x i16> @f15() {
; CHECK-LABEL: f15:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  larl %r1, .LCPI
; CHECK-NEXT:  vl %v0, 0(%r1)
; CHECK-NEXT:  vupllb %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <8 x i16> @llvm.s390.vupllb(<16 x i8>
                                         <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
                                          i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  %and = and <8 x i16> %unp, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %and
}

declare <4 x i32> @llvm.s390.vuplhw(<8 x i16>)
declare <4 x i32> @llvm.s390.vupllh(<8 x i16>)

; VUPLHW (used operand elements are 0)
define <4 x i32> @f16() {
; CHECK-LABEL: f16:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <4 x i32> @llvm.s390.vuplhw(<8 x i16>
                                          <i16 1, i16 1, i16 1, i16 1,
                                           i16 0, i16 0, i16 0, i16 0>)

  %and = and <4 x i32> %unp, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; VUPLHW (used operand elements are 1)
define <4 x i32> @f17() {
; CHECK-LABEL: f17:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  larl %r1, .LCPI
; CHECK-NEXT:  vl %v0, 0(%r1)
; CHECK-NEXT:  vuplhw %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <4 x i32> @llvm.s390.vuplhw(<8 x i16>
                                          <i16 0, i16 0, i16 0, i16 0,
                                           i16 1, i16 1, i16 1, i16 1>)
  %and = and <4 x i32> %unp, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; VUPLLH (used operand elements are 0)
define <4 x i32> @f18() {
; CHECK-LABEL: f18:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <4 x i32> @llvm.s390.vupllh(<8 x i16>
                                          <i16 1, i16 1, i16 1, i16 1,
                                           i16 0, i16 0, i16 0, i16 0>)
  %and = and <4 x i32> %unp, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

; VUPLLH (used operand elements are 1)
define <4 x i32> @f19() {
; CHECK-LABEL: f19:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  larl %r1, .LCPI
; CHECK-NEXT:  vl %v0, 0(%r1)
; CHECK-NEXT:  vupllh %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <4 x i32> @llvm.s390.vupllh(<8 x i16>
                                          <i16 0, i16 0, i16 0, i16 0,
                                           i16 1, i16 1, i16 1, i16 1>)
  %and = and <4 x i32> %unp, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %and
}

declare <2 x i64> @llvm.s390.vuplf(<4 x i32>)
declare <2 x i64> @llvm.s390.vupllf(<4 x i32>)

; VUPLF (used operand elements are 0)
define <2 x i64> @f20() {
; CHECK-LABEL: f20:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <2 x i64> @llvm.s390.vuplf(<4 x i32> <i32 1, i32 1, i32 0, i32 0>)
  %and = and <2 x i64> %unp, <i64 1, i64 1>
  ret <2 x i64> %and
}

; VUPLF (used operand elements are 1)
define <2 x i64> @f21() {
; CHECK-LABEL: f21:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  larl %r1, .LCPI
; CHECK-NEXT:  vl %v0, 0(%r1)
; CHECK-NEXT:  vuplf %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <2 x i64> @llvm.s390.vuplf(<4 x i32> <i32 0, i32 0, i32 1, i32 1>)
  %and = and <2 x i64> %unp, <i64 1, i64 1>
  ret <2 x i64> %and
}

; VUPLLF (used operand elements are 0)
define <2 x i64> @f22() {
; CHECK-LABEL: f22:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <2 x i64> @llvm.s390.vupllf(<4 x i32> <i32 1, i32 1, i32 0, i32 0>)
  %and = and <2 x i64> %unp, <i64 1, i64 1>
  ret <2 x i64> %and
}

; VUPLLF (used operand elements are 1)
define <2 x i64> @f23() {
; CHECK-LABEL: f23:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  larl %r1, .LCPI
; CHECK-NEXT:  vl %v0, 0(%r1)
; CHECK-NEXT:  vupllf %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <2 x i64> @llvm.s390.vupllf(<4 x i32> <i32 0, i32 0, i32 1, i32 1>)
  %and = and <2 x i64> %unp, <i64 1, i64 1>
  ret <2 x i64> %and
}

; Test that signed unpacking of positive elements gives known zeros in high part.
define <2 x i64> @f24() {
; CHECK-LABEL: f24:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <2 x i64> @llvm.s390.vuphf(<4 x i32> <i32 1, i32 1, i32 0, i32 0>)
  %and = and <2 x i64> %unp, <i64 -4294967296, ; = 0xffffffff00000000
                              i64 -4294967296>
  ret <2 x i64> %and
}

; Test that signed unpacking of negative elements gives known ones in high part.
define <2 x i64> @f25() {
; CHECK-LABEL: f25:
; CHECK-LABEL: # %bb.0:
;                         61680 = 0xf0f0
; CHECK-NEXT:  vgbm %v24, 61680
; CHECK-NEXT:  br %r14
  %unp = call <2 x i64> @llvm.s390.vuphf(<4 x i32> <i32 -1, i32 -1, i32 0, i32 0>)
  %and = and <2 x i64> %unp, <i64 -4294967296, ; = 0xffffffff00000000
                              i64 -4294967296>
  ret <2 x i64> %and
}

; Test that logical unpacking of negative elements gives known zeros in high part.
define <2 x i64> @f26() {
; CHECK-LABEL: f26:
; CHECK-LABEL: # %bb.0:
; CHECK-NEXT:  vgbm %v24, 0
; CHECK-NEXT:  br %r14
  %unp = call <2 x i64> @llvm.s390.vuplhf(<4 x i32> <i32 -1, i32 -1, i32 0, i32 0>)
  %and = and <2 x i64> %unp, <i64 -4294967296, ; = 0xffffffff00000000
                              i64 -4294967296>
  ret <2 x i64> %and
}
