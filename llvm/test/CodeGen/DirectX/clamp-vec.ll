; RUN: opt -S  -dxil-intrinsic-expansion  < %s | FileCheck %s

; Make sure dxil operation function calls for clamp are generated for float/int/uint vectors.

; CHECK-LABEL: clamp_half3
define noundef <3 x half> @clamp_half3(<3 x half> noundef %a, <3 x half> noundef %b, <3 x half> noundef %c) {
entry:
  ; CHECK: call <3 x half> @llvm.maxnum.v3f16(<3 x half>  %a, <3 x half>  %b)
  ; CHECK: call <3 x half> @llvm.minnum.v3f16(<3 x half>  %{{.*}}, <3 x half>  %c)
  %dx.clamp = call <3 x half> @llvm.dx.clamp.v3f16(<3 x half> %a, <3 x half> %b, <3 x half> %c)
  ret <3 x half> %dx.clamp
}

; CHECK-LABEL: clamp_float4
define noundef <4 x float> @clamp_float4(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c) {
entry:
  ; CHECK: call <4 x float> @llvm.maxnum.v4f32(<4 x float>  %a, <4 x float>  %b)
  ; CHECK: call <4 x float> @llvm.minnum.v4f32(<4 x float>  %{{.*}}, <4 x float>  %c)
  %dx.clamp = call <4 x float> @llvm.dx.clamp.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %dx.clamp
}

; CHECK-LABEL: clamp_double2
define noundef <2 x double> @clamp_double2(<2 x double> noundef %a, <2 x double> noundef %b, <2 x double> noundef %c) {
entry:
  ; CHECK: call <2 x double> @llvm.maxnum.v2f64(<2 x double>  %a, <2 x double>  %b)
  ; CHECK: call <2 x double> @llvm.minnum.v2f64(<2 x double>  %{{.*}}, <2 x double>  %c)
  %dx.clamp = call <2 x double> @llvm.dx.clamp.v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c)
  ret <2 x double> %dx.clamp
}

; CHECK-LABEL: clamp_int4
define noundef <4 x i32> @clamp_int4(<4 x i32> noundef %a, <4 x i32> noundef %b, <4 x i32> noundef %c) {
entry:
  ; CHECK: call <4 x i32> @llvm.smax.v4i32(<4 x i32> %a, <4 x i32> %b)
  ; CHECK: call <4 x i32> @llvm.smin.v4i32(<4 x i32> %{{.*}}, <4 x i32> %c)
  %dx.clamp = call <4 x i32> @llvm.dx.clamp.v4i32(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c)
  ret <4 x i32> %dx.clamp
}

; CHECK-LABEL: clamp_uint16_t3
define noundef <3 x i16> @clamp_uint16_t3(<3 x i16> noundef %a, <3 x i16> noundef %b, <3 x i16> noundef %c) {
entry:
  ; CHECK: call <3 x i16> @llvm.umax.v3i16(<3 x i16>  %a, <3 x i16>  %b)
  ; CHECK: call <3 x i16> @llvm.umin.v3i16(<3 x i16>  %{{.*}}, <3 x i16>  %c)
  %dx.clamp = call <3 x i16> @llvm.dx.uclamp.v3i16(<3 x i16> %a, <3 x i16> %b, <3 x i16> %c)
  ret <3 x i16> %dx.clamp
}

; CHECK-LABEL: clamp_uint4
define noundef <4 x i32> @clamp_uint4(<4 x i32> noundef %a, <4 x i32> noundef %b, <4 x i32> noundef %c) {
entry:
  ; CHECK: call <4 x i32> @llvm.umax.v4i32(<4 x i32>  %a, <4 x i32>  %b)
  ; CHECK: call <4 x i32> @llvm.umin.v4i32(<4 x i32>  %{{.*}}, <4 x i32>  %c)
  %dx.clamp = call <4 x i32> @llvm.dx.uclamp.v4i32(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c)
  ret <4 x i32> %dx.clamp
}

; CHECK-LABEL: clamp_uint64_t4
define noundef <2 x i64> @clamp_uint64_t4(<2 x i64> noundef %a, <2 x i64> noundef %b, <2 x i64> noundef %c) {
entry:
  ; CHECK: call <2 x i64> @llvm.umax.v2i64(<2 x i64>  %a, <2 x i64>  %b)
  ; CHECK: call <2 x i64> @llvm.umin.v2i64(<2 x i64>  %{{.*}}, <2 x i64>  %c)
  %dx.clamp = call <2 x i64> @llvm.dx.uclamp.v2i64(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c)
  ret <2 x i64> %dx.clamp
}

declare <3 x half> @llvm.dx.clamp.v3f16(<3 x half>, <3 x half>, <3 x half>)
declare <4 x float> @llvm.dx.clamp.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <2 x double> @llvm.dx.clamp.v2f64(<2 x double>, <2 x double>, <2 x double>)
declare <4 x i32> @llvm.dx.clamp.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
declare <3 x i16> @llvm.dx.uclamp.v3i32(<3 x i16>, <3 x i32>, <3 x i16>)
declare <4 x i32> @llvm.dx.uclamp.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
declare <2 x i64> @llvm.dx.uclamp.v2i64(<2 x i64>, <2 x i64>, <2 x i64>)
