; Test vector population-count instruction on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %a)
declare <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %a)
declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %a)
declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %a)

define <16 x i8> @f1(<16 x i8> %a) {
; CHECK-LABEL: f1:
; CHECK: vpopctb  %v24, %v24
; CHECK: br      %r14

  %popcnt = call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %a)
  ret <16 x i8> %popcnt
}

define <8 x i16> @f2(<8 x i16> %a) {
; CHECK-LABEL: f2:
; CHECK: vpopcth  %v24, %v24
; CHECK: br      %r14

  %popcnt = call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %a)
  ret <8 x i16> %popcnt
}

define <4 x i32> @f3(<4 x i32> %a) {
; CHECK-LABEL: f3:
; CHECK: vpopctf  %v24, %v24
; CHECK: br      %r14

  %popcnt = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %a)
  ret <4 x i32> %popcnt
}

define <2 x i64> @f4(<2 x i64> %a) {
; CHECK-LABEL: f4:
; CHECK: vpopctg  %v24, %v24
; CHECK: br      %r14

  %popcnt = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %a)
  ret <2 x i64> %popcnt
}

