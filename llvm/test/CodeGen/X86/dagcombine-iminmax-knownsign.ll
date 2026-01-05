; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+sse2,-sse4.1 < %s | FileCheck %s

declare <16 x i8> @llvm.smax.v16i8(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.smin.v16i8(<16 x i8>, <16 x i8>)

define <16 x i8> @smax_both_negative(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: smax_both_negative:
; CHECK: pmaxub
  %mask0 = insertelement <16 x i8> poison, i8 -128, i64 0
  %mask = shufflevector <16 x i8> %mask0, <16 x i8> poison,
                        <16 x i32> zeroinitializer
  %a1 = or <16 x i8> %a, %mask
  %b1 = or <16 x i8> %b, %mask
  %r = call <16 x i8> @llvm.smax.v16i8(<16 x i8> %a1, <16 x i8> %b1)
  ret <16 x i8> %r
}

define <16 x i8> @smin_both_negative(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: smin_both_negative:
; CHECK: pminub
  %mask0 = insertelement <16 x i8> poison, i8 -128, i64 0
  %mask = shufflevector <16 x i8> %mask0, <16 x i8> poison,
                        <16 x i32> zeroinitializer
  %a1 = or <16 x i8> %a, %mask
  %b1 = or <16 x i8> %b, %mask
  %r = call <16 x i8> @llvm.smin.v16i8(<16 x i8> %a1, <16 x i8> %b1)
  ret <16 x i8> %r
}

define <16 x i8> @smax_both_nonnegative(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: smax_both_nonnegative:
; CHECK: pmaxub
  %mask0 = insertelement <16 x i8> poison, i8 127, i64 0
  %mask = shufflevector <16 x i8> %mask0, <16 x i8> poison,
                        <16 x i32> zeroinitializer
  %a1 = and <16 x i8> %a, %mask
  %b1 = and <16 x i8> %b, %mask
  %r = call <16 x i8> @llvm.smax.v16i8(<16 x i8> %a1, <16 x i8> %b1)
  ret <16 x i8> %r
}

define <16 x i8> @smin_both_nonnegative(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: smin_both_nonnegative:
; CHECK: pminub
  %mask0 = insertelement <16 x i8> poison, i8 127, i64 0
  %mask = shufflevector <16 x i8> %mask0, <16 x i8> poison,
                        <16 x i32> zeroinitializer
  %a1 = and <16 x i8> %a, %mask
  %b1 = and <16 x i8> %b, %mask
  %r = call <16 x i8> @llvm.smin.v16i8(<16 x i8> %a1, <16 x i8> %b1)
  ret <16 x i8> %r
}
