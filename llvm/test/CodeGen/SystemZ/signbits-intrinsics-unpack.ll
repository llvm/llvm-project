; Test that DAGCombiner gets helped by ComputeNumSignBitsForTargetNode() with
; vector intrinsics.
;
; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 < %s  | FileCheck %s

declare <8 x i16> @llvm.s390.vuphb(<16 x i8>)

; VUPHB
define <8 x i16> @f0() {
; CHECK-LABEL: f0:
; CHECK-LABEL: # %bb.0:
; CHECK:       vuphb %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <8 x i16> @llvm.s390.vuphb(<16 x i8>
                                         <i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1,
                                          i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1>)
  %trunc = trunc <8 x i16> %unp to <8 x i8>
  %ret = sext <8 x i8> %trunc to <8 x i16>
  ret <8 x i16> %ret
}

declare <4 x i32> @llvm.s390.vuphh(<8 x i16>)

; VUPHH
define <4 x i32> @f1() {
; CHECK-LABEL: f1:
; CHECK-LABEL: # %bb.0:
; CHECK:       vuphh %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <4 x i32> @llvm.s390.vuphh(<8 x i16>
                                         <i16 0, i16 1, i16 0, i16 1,
                                          i16 0, i16 1, i16 0, i16 1>)
  %trunc = trunc <4 x i32> %unp to <4 x i16>
  %ret = sext <4 x i16> %trunc to <4 x i32>
  ret <4 x i32> %ret
}

declare <2 x i64> @llvm.s390.vuphf(<4 x i32>)

; VUPHF
define <2 x i64> @f2() {
; CHECK-LABEL: f2:
; CHECK-LABEL: # %bb.0:
; CHECK:       vuphf %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <2 x i64> @llvm.s390.vuphf(<4 x i32> <i32 0, i32 1, i32 0, i32 1>)
  %trunc = trunc <2 x i64> %unp to <2 x i32>
  %ret = sext <2 x i32> %trunc to <2 x i64>
  ret <2 x i64> %ret
}

declare <8 x i16> @llvm.s390.vuplb(<16 x i8>)

; VUPLB
define <8 x i16> @f3() {
; CHECK-LABEL: f3:
; CHECK-LABEL: # %bb.0:
; CHECK:       vuplb %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <8 x i16> @llvm.s390.vuplb(<16 x i8>
                                         <i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1,
                                          i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1>)
  %trunc = trunc <8 x i16> %unp to <8 x i8>
  %ret = sext <8 x i8> %trunc to <8 x i16>
  ret <8 x i16> %ret
}

declare <4 x i32> @llvm.s390.vuplhw(<8 x i16>)

; VUPLHW
define <4 x i32> @f4() {
; CHECK-LABEL: f4:
; CHECK-LABEL: # %bb.0:
; CHECK:       vuplhw %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <4 x i32> @llvm.s390.vuplhw(<8 x i16>
                                          <i16 1, i16 0, i16 1, i16 0,
                                           i16 1, i16 0, i16 1, i16 0>)
  %trunc = trunc <4 x i32> %unp to <4 x i16>
  %ret = sext <4 x i16> %trunc to <4 x i32>
  ret <4 x i32> %ret
}

declare <2 x i64> @llvm.s390.vuplf(<4 x i32>)

; VUPLF
define <2 x i64> @f5() {
; CHECK-LABEL: f5:
; CHECK-LABEL: # %bb.0:
; CHECK:       vuplf %v24, %v0
; CHECK-NEXT:  br %r14
  %unp = call <2 x i64> @llvm.s390.vuplf(<4 x i32> <i32 1, i32 0, i32 1, i32 0>)
  %trunc = trunc <2 x i64> %unp to <2 x i32>
  %ret = sext <2 x i32> %trunc to <2 x i64>
  ret <2 x i64> %ret
}

