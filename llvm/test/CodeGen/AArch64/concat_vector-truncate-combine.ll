; RUN: llc < %s -mtriple arm64-apple-darwin -asm-verbose=false | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; Test the (concat_vectors (trunc), (trunc)) pattern.

define <4 x i16> @test_concat_truncate_v2i64_to_v4i16(<2 x i64> %a, <2 x i64> %b) #0 {
entry:
; CHECK-LABEL: test_concat_truncate_v2i64_to_v4i16:
; CHECK-NEXT: uzp1.4s v0, v0, v1
; CHECK-NEXT: xtn.4h v0, v0
; CHECK-NEXT: ret
  %at = trunc <2 x i64> %a to <2 x i16>
  %bt = trunc <2 x i64> %b to <2 x i16>
  %shuffle = shufflevector <2 x i16> %at, <2 x i16> %bt, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i16> %shuffle
}

define <8 x i8> @test_concat_truncate_v4i32_to_v8i8(<4 x i32> %a, <4 x i32> %b) #0 {
entry:
; CHECK-LABEL: test_concat_truncate_v4i32_to_v8i8:
; CHECK-NEXT: uzp1.8h v0, v0, v1
; CHECK-NEXT: xtn.8b v0, v0
; CHECK-NEXT: ret
  %at = trunc <4 x i32> %a to <4 x i8>
  %bt = trunc <4 x i32> %b to <4 x i8>
  %shuffle = shufflevector <4 x i8> %at, <4 x i8> %bt, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i8> %shuffle
}

define <8 x i16> @test_concat_truncate_v4i32_to_v8i16(<4 x i32> %a, <4 x i32> %b) #0 {
entry:
; CHECK-LABEL: test_concat_truncate_v4i32_to_v8i16:
; CHECK-NEXT: xtn.4h v0, v0
; CHECK-NEXT: xtn2.8h v0, v1
; CHECK-NEXT: ret
  %at = trunc <4 x i32> %a to <4 x i16>
  %bt = trunc <4 x i32> %b to <4 x i16>
  %shuffle = shufflevector <4 x i16> %at, <4 x i16> %bt, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %shuffle
}


; The concat_vectors operation in this test is introduced when splitting
; the fptrunc operation due to the split <vscale x 4 x double> input operand.
define void @test_concat_fptrunc_v4f64_to_v4f32(<vscale x 4 x float>* %ptr) #1 {
entry:
; CHECK-LABEL: test_concat_fptrunc_v4f64_to_v4f32:
; CHECK:       fmov    z0.d, #1.00000000
; CHECK-NEXT:  ptrue   p0.d
; CHECK-NEXT:  fcvt    z0.s, p0/m, z0.d
; CHECK-NEXT:  ptrue   p0.s
; CHECK-NEXT:  uzp1    z0.s, z0.s, z0.s
; CHECK-NEXT:  st1w    { z0.s }, p0, [x0]
; CHECK-NEXT:  ret
  %0 = shufflevector <vscale x 4 x double> insertelement (<vscale x 4 x double> poison, double 1.000000e+00, i32 0), <vscale x 4 x double> poison, <vscale x 4 x i32> zeroinitializer
  %1 = fptrunc <vscale x 4 x double> %0 to <vscale x 4 x float>
  store <vscale x 4 x float> %1, <vscale x 4 x float>* %ptr, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { "target-features"="+sve" }
