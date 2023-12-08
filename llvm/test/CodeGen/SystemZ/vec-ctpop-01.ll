; Test vector population-count instruction
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %a)
declare <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %a)
declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %a)
declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %a)

define <16 x i8> @f1(<16 x i8> %a) {
; CHECK-LABEL: f1:
; CHECK: vpopct  %v24, %v24, 0
; CHECK: br      %r14

  %popcnt = call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %a)
  ret <16 x i8> %popcnt
}

define <8 x i16> @f2(<8 x i16> %a) {
; CHECK-LABEL: f2:
; CHECK: vpopct  [[T1:%v[0-9]+]], %v24, 0
; CHECK: veslh   [[T2:%v[0-9]+]], [[T1]], 8
; CHECK: vah     [[T3:%v[0-9]+]], [[T1]], [[T2]]
; CHECK: vesrlh  %v24, [[T3]], 8
; CHECK: br      %r14

  %popcnt = call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %a)
  ret <8 x i16> %popcnt
}

define <4 x i32> @f3(<4 x i32> %a) {
; CHECK-LABEL: f3:
; CHECK: vpopct  [[T1:%v[0-9]+]], %v24, 0
; CHECK: vgbm    [[T2:%v[0-9]+]], 0
; CHECK: vsumb   %v24, [[T1]], [[T2]]
; CHECK: br      %r14

  %popcnt = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %a)
  ret <4 x i32> %popcnt
}

define <2 x i64> @f4(<2 x i64> %a) {
; CHECK-LABEL: f4:
; CHECK: vpopct  [[T1:%v[0-9]+]], %v24, 0
; CHECK: vgbm    [[T2:%v[0-9]+]], 0
; CHECK: vsumb   [[T3:%v[0-9]+]], [[T1]], [[T2]]
; CHECK: vsumgf  %v24, [[T3]], [[T2]]
; CHECK: br      %r14

  %popcnt = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %a)
  ret <2 x i64> %popcnt
}

