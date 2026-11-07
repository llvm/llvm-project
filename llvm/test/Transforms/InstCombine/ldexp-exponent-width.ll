; RUN: opt -passes=instcombine -S %s | FileCheck %s

declare half @llvm.ldexp.f16.i8(half, i8)
declare float @llvm.ldexp.f32.i32(float, i32)
declare double @llvm.ldexp.f64.i8(double, i8)
declare double @llvm.ldexp.f64.i64(double, i64)
declare double @llvm.ldexp.f64.i128(double, i128)
declare <2 x double> @llvm.ldexp.v2f64.v2i8(<2 x double>, <2 x i8>)

define double @narrow_negative(double %x) {
; CHECK-LABEL: @narrow_negative(
; CHECK: fmul double %x, 5.000000e-01
  %r = call double @llvm.ldexp.f64.i8(double %x, i8 -1)
  ret double %r
}

define double @narrow_positive(double %x) {
; CHECK-LABEL: @narrow_positive(
; CHECK: fmul double %x, 8.000000e+00
  %r = call double @llvm.ldexp.f64.i8(double %x, i8 3)
  ret double %r
}

define double @wide_positive(double %x) {
; CHECK-LABEL: @wide_positive(
; CHECK: call double @llvm.ldexp.f64.i64(double %x, i64 4294967296)
  %r = call double @llvm.ldexp.f64.i64(double %x, i64 4294967296)
  ret double %r
}

define double @wide_negative(double %x) {
; CHECK-LABEL: @wide_negative(
; CHECK: call double @llvm.ldexp.f64.i64(double %x, i64 -4294967296)
  %r = call double @llvm.ldexp.f64.i64(double %x, i64 -4294967296)
  ret double %r
}

define double @very_wide(double %x) {
; CHECK-LABEL: @very_wide(
; CHECK: call double @llvm.ldexp.f64.i128(double %x, i128 18446744073709551616)
  %r = call double @llvm.ldexp.f64.i128(double %x, i128 18446744073709551616)
  ret double %r
}

define <2 x double> @narrow_splat(<2 x double> %x) {
; CHECK-LABEL: @narrow_splat(
; CHECK: fmul <2 x double> %x, splat (double 5.000000e-01)
  %r = call <2 x double> @llvm.ldexp.v2f64.v2i8(<2 x double> %x, <2 x i8> splat (i8 -1))
  ret <2 x double> %r
}

define double @narrow_minimum(double %x) {
; CHECK-LABEL: @narrow_minimum(
; CHECK: fmul double %x, f0x37F0000000000000
  %r = call double @llvm.ldexp.f64.i8(double %x, i8 -128)
  ret double %r
}

define double @very_wide_small(double %x) {
; CHECK-LABEL: @very_wide_small(
; CHECK: fmul double %x, 5.000000e-01
  %r = call double @llvm.ldexp.f64.i128(double %x, i128 -1)
  ret double %r
}

; A subnormal scale must not become an fmul input when it may be flushed.
define half @narrow_subnormal_flush(half %x) denormal_fpenv(preservesign) {
; CHECK-LABEL: @narrow_subnormal_flush(
; CHECK: call half @llvm.ldexp.f16.i8(half %x, i8 -15)
  %r = call half @llvm.ldexp.f16.i8(half %x, i8 -15)
  ret half %r
}

define half @narrow_subnormal_ieee(half %x) {
; CHECK-LABEL: @narrow_subnormal_ieee(
; CHECK: fmul half %x,
  %r = call half @llvm.ldexp.f16.i8(half %x, i8 -15)
  ret half %r
}

define float @subnormal_dynamic(float %x) denormal_fpenv(dynamic) {
; CHECK-LABEL: @subnormal_dynamic(
; CHECK: call float @llvm.ldexp.f32.i32(float %x, i32 -127)
  %r = call float @llvm.ldexp.f32.i32(float %x, i32 -127)
  ret float %r
}

define float @normal_flush(float %x) denormal_fpenv(preservesign) {
; CHECK-LABEL: @normal_flush(
; CHECK: fmul float %x,
  %r = call float @llvm.ldexp.f32.i32(float %x, i32 -126)
  ret float %r
}

define float @subnormal_output_flush_only(float %x) denormal_fpenv(preservesign|ieee) {
; CHECK-LABEL: @subnormal_output_flush_only(
; CHECK: fmul float %x,
  %r = call float @llvm.ldexp.f32.i32(float %x, i32 -127)
  ret float %r
}

define float @subnormal_positive_zero(float %x) denormal_fpenv(positivezero) {
; CHECK-LABEL: @subnormal_positive_zero(
; CHECK: call float @llvm.ldexp.f32.i32(float %x, i32 -127)
  %r = call float @llvm.ldexp.f32.i32(float %x, i32 -127)
  ret float %r
}
