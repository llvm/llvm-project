; Test vector count trailing zeros
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare <16 x i8> @llvm.cttz.v16i8(<16 x i8> %src, i1 %is_zero_undef)
declare <8 x i16> @llvm.cttz.v8i16(<8 x i16> %src, i1 %is_zero_undef)
declare <4 x i32> @llvm.cttz.v4i32(<4 x i32> %src, i1 %is_zero_undef)
declare <2 x i64> @llvm.cttz.v2i64(<2 x i64> %src, i1 %is_zero_undef)

define <16 x i8> @f1(<16 x i8> %a) {
; CHECK-LABEL: f1:
; CHECK: vctzb %v24, %v24
; CHECK: br    %r14

  %res = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %a, i1 false)
  ret <16 x i8> %res
}

define <16 x i8> @f2(<16 x i8> %a) {
; CHECK-LABEL: f2:
; CHECK: vctzb %v24, %v24
; CHECK: br    %r14

  %res = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %a, i1 true)
  ret <16 x i8> %res
}

define <8 x i16> @f3(<8 x i16> %a) {
; CHECK-LABEL: f3:
; CHECK: vctzh %v24, %v24
; CHECK: br    %r14

  %res = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %a, i1 false)
  ret <8 x i16> %res
}

define <8 x i16> @f4(<8 x i16> %a) {
; CHECK-LABEL: f4:
; CHECK: vctzh %v24, %v24
; CHECK: br    %r14

  %res = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %a, i1 true)
  ret <8 x i16> %res
}

define <4 x i32> @f5(<4 x i32> %a) {
; CHECK-LABEL: f5:
; CHECK: vctzf %v24, %v24
; CHECK: br    %r14

  %res = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %a, i1 false)
  ret <4 x i32> %res
}

define <4 x i32> @f6(<4 x i32> %a) {
; CHECK-LABEL: f6:
; CHECK: vctzf %v24, %v24
; CHECK: br    %r14

  %res = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %a, i1 true)
  ret <4 x i32> %res
}

define <2 x i64> @f7(<2 x i64> %a) {
; CHECK-LABEL: f7:
; CHECK: vctzg %v24, %v24
; CHECK: br    %r14

  %res = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %a, i1 false)
  ret <2 x i64> %res
}

define <2 x i64> @f8(<2 x i64> %a) {
; CHECK-LABEL: f8:
; CHECK: vctzg %v24, %v24
; CHECK: br    %r14

  %res = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %a, i1 true)
  ret <2 x i64> %res
}

