; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; Test valid scalar and vector conversions for arbitrary FP intrinsics

declare i8 @llvm.convert.to.arbitrary.fp.i8.f16(half, metadata, metadata, i1)
declare <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4f16(<4 x half>, metadata, metadata, i1)
declare <8 x i8> @llvm.convert.to.arbitrary.fp.v8i8.v8f16(<8 x half>, metadata, metadata, i1)
declare <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4f32(<4 x float>, metadata, metadata, i1)

declare half @llvm.convert.from.arbitrary.fp.f16.i8(i8, metadata)
declare <4 x half> @llvm.convert.from.arbitrary.fp.v4f16.v4i8(<4 x i8>, metadata)
declare <8 x half> @llvm.convert.from.arbitrary.fp.v8f16.v8i8(<8 x i8>, metadata)
declare float @llvm.convert.from.arbitrary.fp.f32.i8(i8, metadata)
declare <4 x float> @llvm.convert.from.arbitrary.fp.v4f32.v4i8(<4 x i8>, metadata)

; Scalar conversions to arbitrary FP
; CHECK-LABEL: @good_half_to_fp8(
define i8 @good_half_to_fp8(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"Float8E5M2", metadata !"round.towardzero", i1 true)
  ret i8 %r
}

; CHECK-LABEL: @good_half_to_fp8_fnuz(
define i8 @good_half_to_fp8_fnuz(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"Float8E4M3FNUZ", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

; CHECK-LABEL: @good_half_to_fp8_fn(
define i8 @good_half_to_fp8_fn(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"Float8E4M3FN", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

; Scalar conversions from arbitrary FP
; CHECK-LABEL: @good_fp8_to_half(
define half @good_fp8_to_half(i8 %v) {
  %r = call half @llvm.convert.from.arbitrary.fp.f16.i8(
      i8 %v, metadata !"Float8E4M3")
  ret half %r
}

; CHECK-LABEL: @good_fp8_e5m2_to_half(
define half @good_fp8_e5m2_to_half(i8 %v) {
  %r = call half @llvm.convert.from.arbitrary.fp.f16.i8(
      i8 %v, metadata !"Float8E5M2")
  ret half %r
}

; CHECK-LABEL: @good_fp8_to_float(
define float @good_fp8_to_float(i8 %v) {
  %r = call float @llvm.convert.from.arbitrary.fp.f32.i8(
      i8 %v, metadata !"Float8E4M3")
  ret float %r
}

; Vector conversions to arbitrary FP
; CHECK-LABEL: @good_vec4_half_to_fp8(
define <4 x i8> @good_vec4_half_to_fp8(<4 x half> %v) {
  %r = call <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4f16(
      <4 x half> %v, metadata !"Float8E4M3FN", metadata !"round.towardzero", i1 true)
  ret <4 x i8> %r
}

; CHECK-LABEL: @good_vec8_half_to_fp8(
define <8 x i8> @good_vec8_half_to_fp8(<8 x half> %v) {
  %r = call <8 x i8> @llvm.convert.to.arbitrary.fp.v8i8.v8f16(
      <8 x half> %v, metadata !"Float8E5M2FNUZ", metadata !"round.tonearest", i1 false)
  ret <8 x i8> %r
}

; CHECK-LABEL: @good_vec4_float_to_fp8(
define <4 x i8> @good_vec4_float_to_fp8(<4 x float> %v) {
  %r = call <4 x i8> @llvm.convert.to.arbitrary.fp.v4i8.v4f32(
      <4 x float> %v, metadata !"Float8E4M3B11FNUZ", metadata !"round.tonearest", i1 false)
  ret <4 x i8> %r
}

; Vector conversions from arbitrary FP
; CHECK-LABEL: @good_vec4_fp8_to_half(
define <4 x half> @good_vec4_fp8_to_half(<4 x i8> %v) {
  %r = call <4 x half> @llvm.convert.from.arbitrary.fp.v4f16.v4i8(
      <4 x i8> %v, metadata !"Float8E4M3")
  ret <4 x half> %r
}

; CHECK-LABEL: @good_vec8_fp8_to_half(
define <8 x half> @good_vec8_fp8_to_half(<8 x i8> %v) {
  %r = call <8 x half> @llvm.convert.from.arbitrary.fp.v8f16.v8i8(
      <8 x i8> %v, metadata !"Float8E5M2")
  ret <8 x half> %r
}

; CHECK-LABEL: @good_vec4_fp8_to_float(
define <4 x float> @good_vec4_fp8_to_float(<4 x i8> %v) {
  %r = call <4 x float> @llvm.convert.from.arbitrary.fp.v4f32.v4i8(
      <4 x i8> %v, metadata !"Float8E4M3B11FNUZ")
  ret <4 x float> %r
}

; Test different rounding modes
; CHECK-LABEL: @good_rounding_towardzero(
define i8 @good_rounding_towardzero(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"Float8E4M3", metadata !"round.towardzero", i1 false)
  ret i8 %r
}

; CHECK-LABEL: @good_rounding_upward(
define i8 @good_rounding_upward(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"Float8E4M3", metadata !"round.upward", i1 false)
  ret i8 %r
}

; CHECK-LABEL: @good_rounding_downward(
define i8 @good_rounding_downward(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"Float8E4M3", metadata !"round.downward", i1 false)
  ret i8 %r
}

; Test all supported formats
; CHECK-LABEL: @good_float8_e5m2_fnuz(
define i8 @good_float8_e5m2_fnuz(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"Float8E5M2FNUZ", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

; CHECK-LABEL: @good_float8_e3m4(
define i8 @good_float8_e3m4(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"Float8E3M4", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

; CHECK-LABEL: @good_float8_e8m0fnu(
define i8 @good_float8_e8m0fnu(half %v) {
  %r = call i8 @llvm.convert.to.arbitrary.fp.i8.f16(
      half %v, metadata !"Float8E8M0FNU", metadata !"round.tonearest", i1 false)
  ret i8 %r
}

; CHECK-LABEL: @good_float6_e3m2fn(
define i6 @good_float6_e3m2fn(half %v) {
  %r = call i6 @llvm.convert.to.arbitrary.fp.i6.f16(
      half %v, metadata !"Float6E3M2FN", metadata !"round.tonearest", i1 false)
  ret i6 %r
}

; CHECK-LABEL: @good_float6_e2m3fn(
define i6 @good_float6_e2m3fn(half %v) {
  %r = call i6 @llvm.convert.to.arbitrary.fp.i6.f16(
      half %v, metadata !"Float6E2M3FN", metadata !"round.tonearest", i1 false)
  ret i6 %r
}

; CHECK-LABEL: @good_float4_e2m1fn(
define i4 @good_float4_e2m1fn(half %v) {
  %r = call i4 @llvm.convert.to.arbitrary.fp.i4.f16(
      half %v, metadata !"Float4E2M1FN", metadata !"round.tonearest", i1 false)
  ret i4 %r
}

declare i6 @llvm.convert.to.arbitrary.fp.i6.f16(half, metadata, metadata, i1)
declare i4 @llvm.convert.to.arbitrary.fp.i4.f16(half, metadata, metadata, i1)
