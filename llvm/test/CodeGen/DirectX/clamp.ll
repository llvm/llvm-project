; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for clamp/uclamp are generated for half/float/double/i16/i32/i64.

; CHECK-LABEL:test_clamp_i16
define noundef i16 @test_clamp_i16(i16 noundef %a, i16 noundef %b, i16 noundef %c) {
entry:
; CHECK: call i16 @dx.op.binary.i16(i32 37, i16 %{{.*}}, i16 %{{.*}})
; CHECK: call i16 @dx.op.binary.i16(i32 38, i16 %{{.*}}, i16 %{{.*}})
  %0 = call i16 @llvm.dx.sclamp.i16(i16 %a, i16 %b, i16 %c)
  ret i16 %0
}

; CHECK-LABEL:test_clamp_i32
define noundef i32 @test_clamp_i32(i32 noundef %a, i32 noundef %b, i32 noundef %c) {
entry:
; CHECK: call i32 @dx.op.binary.i32(i32 37, i32 %{{.*}}, i32 %{{.*}})
; CHECK: call i32 @dx.op.binary.i32(i32 38, i32 %{{.*}}, i32 %{{.*}})
  %0 = call i32 @llvm.dx.sclamp.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %0
}

; CHECK-LABEL:test_clamp_i64
define noundef i64 @test_clamp_i64(i64 noundef %a, i64 noundef %b, i64 noundef %c) {
entry:
; CHECK: call i64 @dx.op.binary.i64(i32 37, i64 %a, i64 %b)
; CHECK: call i64 @dx.op.binary.i64(i32 38, i64 %{{.*}}, i64 %c)
  %0 = call i64 @llvm.dx.sclamp.i64(i64 %a, i64 %b, i64 %c)
  ret i64 %0
}

; CHECK-LABEL:test_clamp_half
define noundef half @test_clamp_half(half noundef %a, half noundef %b, half noundef %c) {
entry:
; CHECK: call half @dx.op.binary.f16(i32 35, half %{{.*}}, half %{{.*}})
; CHECK: call half @dx.op.binary.f16(i32 36, half %{{.*}}, half %{{.*}})
  %0 = call half @llvm.dx.nclamp.f16(half %a, half %b, half %c)
  ret half %0
}

; CHECK-LABEL:test_clamp_float
define noundef float @test_clamp_float(float noundef %a, float noundef %b, float noundef %c) {
entry:
; CHECK: call float @dx.op.binary.f32(i32 35, float %{{.*}}, float %{{.*}})
; CHECK: call float @dx.op.binary.f32(i32 36, float %{{.*}}, float %{{.*}})
  %0 = call float @llvm.dx.nclamp.f32(float %a, float %b, float %c)
  ret float %0
}

; CHECK-LABEL:test_clamp_double
define noundef double @test_clamp_double(double noundef %a, double noundef %b, double noundef %c) {
entry:
; CHECK: call double @dx.op.binary.f64(i32 35, double %{{.*}}, double %{{.*}})
; CHECK: call double @dx.op.binary.f64(i32 36, double %{{.*}}, double %{{.*}})
  %0 = call double @llvm.dx.nclamp.f64(double %a, double %b, double %c)
  ret double %0
}

; CHECK-LABEL:test_uclamp_i16
define noundef i16 @test_uclamp_i16(i16 noundef %a, i16 noundef %b, i16 noundef %c) {
entry:
; CHECK: call i16 @dx.op.binary.i16(i32 39, i16 %{{.*}}, i16 %{{.*}})
; CHECK: call i16 @dx.op.binary.i16(i32 40, i16 %{{.*}}, i16 %{{.*}})
  %0 = call i16 @llvm.dx.uclamp.i16(i16 %a, i16 %b, i16 %c)
  ret i16 %0
}

; CHECK-LABEL:test_uclamp_i32
define noundef i32 @test_uclamp_i32(i32 noundef %a, i32 noundef %b, i32 noundef %c) {
entry:
; CHECK: call i32 @dx.op.binary.i32(i32 39, i32 %{{.*}}, i32 %{{.*}})
; CHECK: call i32 @dx.op.binary.i32(i32 40, i32 %{{.*}}, i32 %{{.*}})
  %0 = call i32 @llvm.dx.uclamp.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %0
}

; CHECK-LABEL:test_uclamp_i64
define noundef i64 @test_uclamp_i64(i64 noundef %a, i64 noundef %b, i64 noundef %c) {
entry:
; CHECK: call i64 @dx.op.binary.i64(i32 39, i64 %a, i64 %b)
; CHECK: call i64 @dx.op.binary.i64(i32 40, i64 %{{.*}}, i64 %c)
  %0 = call i64 @llvm.dx.uclamp.i64(i64 %a, i64 %b, i64 %c)
  ret i64 %0
}

declare half @llvm.dx.nclamp.f16(half, half, half)
declare float @llvm.dx.nclamp.f32(float, float, float)
declare double @llvm.dx.nclamp.f64(double, double, double)
declare i16 @llvm.dx.sclamp.i16(i16, i16, i16)
declare i32 @llvm.dx.sclamp.i32(i32, i32, i32)
declare i64 @llvm.dx.sclamp.i64(i64, i64, i64)
declare i16 @llvm.dx.uclamp.i16(i16, i16, i16)
declare i32 @llvm.dx.uclamp.i32(i32, i32, i32)
declare i64 @llvm.dx.uclamp.i64(i64, i64, i64)

; CHECK-LABEL: clamp_half3
define noundef <3 x half> @clamp_half3(<3 x half> noundef %a, <3 x half> noundef %b, <3 x half> noundef %c) {
entry:
  ; CHECK-DAG: %[[a0:.+]] = extractelement <3 x half> %a, i64 0
  ; CHECK-DAG: %[[a1:.+]] = extractelement <3 x half> %a, i64 1
  ; CHECK-DAG: %[[a2:.+]] = extractelement <3 x half> %a, i64 2
  ; CHECK-DAG: %[[b0:.+]] = extractelement <3 x half> %b, i64 0
  ; CHECK-DAG: %[[b1:.+]] = extractelement <3 x half> %b, i64 1
  ; CHECK-DAG: %[[b2:.+]] = extractelement <3 x half> %b, i64 2
  ; CHECK-DAG: %[[c0:.+]] = extractelement <3 x half> %c, i64 0
  ; CHECK-DAG: %[[c1:.+]] = extractelement <3 x half> %c, i64 1
  ; CHECK-DAG: %[[c2:.+]] = extractelement <3 x half> %c, i64 2
  ; CHECK-DAG: %[[max0:.+]] = call half @dx.op.binary.f16(i32 35, half %[[a0]], half %[[b0]])
  ; CHECK-DAG: %[[max1:.+]] = call half @dx.op.binary.f16(i32 35, half %[[a1]], half %[[b1]])
  ; CHECK-DAG: %[[max2:.+]] = call half @dx.op.binary.f16(i32 35, half %[[a2]], half %[[b2]])
  ; CHECK-DAG: %[[min0:.+]] = call half @dx.op.binary.f16(i32 36, half %[[max0]], half %[[c0]])
  ; CHECK-DAG: %[[min1:.+]] = call half @dx.op.binary.f16(i32 36, half %[[max1]], half %[[c1]])
  ; CHECK-DAG: %[[min2:.+]] = call half @dx.op.binary.f16(i32 36, half %[[max2]], half %[[c2]])
  ; CHECK-DAG: %[[ret0:.+]] = insertelement <3 x half> poison, half %[[min0]], i64 0
  ; CHECK-DAG: %[[ret1:.+]] = insertelement <3 x half> %[[ret0]], half %[[min1]], i64 1
  ; CHECK-DAG: %[[ret2:.+]] = insertelement <3 x half> %[[ret1]], half %[[min2]], i64 2
  ; CHECK: ret <3 x half> %[[ret2]]
  %dx.clamp = call <3 x half> @llvm.dx.nclamp.v3f16(<3 x half> %a, <3 x half> %b, <3 x half> %c)
  ret <3 x half> %dx.clamp
}

; CHECK-LABEL: clamp_float4
define noundef <4 x float> @clamp_float4(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c) {
entry:
  ; CHECK-DAG: %[[a0:.+]] = extractelement <4 x float> %a, i64 0
  ; CHECK-DAG: %[[a1:.+]] = extractelement <4 x float> %a, i64 1
  ; CHECK-DAG: %[[a2:.+]] = extractelement <4 x float> %a, i64 2
  ; CHECK-DAG: %[[a3:.+]] = extractelement <4 x float> %a, i64 3
  ; CHECK-DAG: %[[b0:.+]] = extractelement <4 x float> %b, i64 0
  ; CHECK-DAG: %[[b1:.+]] = extractelement <4 x float> %b, i64 1
  ; CHECK-DAG: %[[b2:.+]] = extractelement <4 x float> %b, i64 2
  ; CHECK-DAG: %[[b3:.+]] = extractelement <4 x float> %b, i64 3
  ; CHECK-DAG: %[[c0:.+]] = extractelement <4 x float> %c, i64 0
  ; CHECK-DAG: %[[c1:.+]] = extractelement <4 x float> %c, i64 1
  ; CHECK-DAG: %[[c2:.+]] = extractelement <4 x float> %c, i64 2
  ; CHECK-DAG: %[[c3:.+]] = extractelement <4 x float> %c, i64 3
  ; CHECK-DAG: %[[max0:.+]] = call float @dx.op.binary.f32(i32 35, float %[[a0]], float %[[b0]])
  ; CHECK-DAG: %[[max1:.+]] = call float @dx.op.binary.f32(i32 35, float %[[a1]], float %[[b1]])
  ; CHECK-DAG: %[[max2:.+]] = call float @dx.op.binary.f32(i32 35, float %[[a2]], float %[[b2]])
  ; CHECK-DAG: %[[max3:.+]] = call float @dx.op.binary.f32(i32 35, float %[[a3]], float %[[b3]])
  ; CHECK-DAG: %[[min0:.+]] = call float @dx.op.binary.f32(i32 36, float %[[max0]], float %[[c0]])
  ; CHECK-DAG: %[[min1:.+]] = call float @dx.op.binary.f32(i32 36, float %[[max1]], float %[[c1]])
  ; CHECK-DAG: %[[min2:.+]] = call float @dx.op.binary.f32(i32 36, float %[[max2]], float %[[c2]])
  ; CHECK-DAG: %[[min3:.+]] = call float @dx.op.binary.f32(i32 36, float %[[max3]], float %[[c3]])
  ; CHECK-DAG: %[[ret0:.+]] = insertelement <4 x float> poison, float %[[min0]], i64 0
  ; CHECK-DAG: %[[ret1:.+]] = insertelement <4 x float> %[[ret0]], float %[[min1]], i64 1
  ; CHECK-DAG: %[[ret2:.+]] = insertelement <4 x float> %[[ret1]], float %[[min2]], i64 2
  ; CHECK-DAG: %[[ret3:.+]] = insertelement <4 x float> %[[ret2]], float %[[min3]], i64 3
  ; CHECK: ret <4 x float> %[[ret3]]
  %dx.clamp = call <4 x float> @llvm.dx.nclamp.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %dx.clamp
}

; CHECK-LABEL: clamp_double2
define noundef <2 x double> @clamp_double2(<2 x double> noundef %a, <2 x double> noundef %b, <2 x double> noundef %c) {
entry:
  ; CHECK-DAG: %[[a0:.+]] = extractelement <2 x double> %a, i64 0
  ; CHECK-DAG: %[[a1:.+]] = extractelement <2 x double> %a, i64 1
  ; CHECK-DAG: %[[b0:.+]] = extractelement <2 x double> %b, i64 0
  ; CHECK-DAG: %[[b1:.+]] = extractelement <2 x double> %b, i64 1
  ; CHECK-DAG: %[[c0:.+]] = extractelement <2 x double> %c, i64 0
  ; CHECK-DAG: %[[c1:.+]] = extractelement <2 x double> %c, i64 1
  ; CHECK-DAG: %[[max0:.+]] = call double @dx.op.binary.f64(i32 35, double %[[a0]], double %[[b0]])
  ; CHECK-DAG: %[[max1:.+]] = call double @dx.op.binary.f64(i32 35, double %[[a1]], double %[[b1]])
  ; CHECK-DAG: %[[min0:.+]] = call double @dx.op.binary.f64(i32 36, double %[[max0]], double %[[c0]])
  ; CHECK-DAG: %[[min1:.+]] = call double @dx.op.binary.f64(i32 36, double %[[max1]], double %[[c1]])
  ; CHECK-DAG: %[[ret0:.+]] = insertelement <2 x double> poison, double %[[min0]], i64 0
  ; CHECK-DAG: %[[ret1:.+]] = insertelement <2 x double> %[[ret0]], double %[[min1]], i64 1
  ; CHECK: ret <2 x double> %[[ret1]]
  %dx.clamp = call <2 x double> @llvm.dx.nclamp.v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c)
  ret <2 x double> %dx.clamp
}

; CHECK-LABEL: clamp_int4
define noundef <4 x i32> @clamp_int4(<4 x i32> noundef %a, <4 x i32> noundef %b, <4 x i32> noundef %c) {
entry:
  ; CHECK-DAG: %[[a0:.+]] = extractelement <4 x i32> %a, i64 0
  ; CHECK-DAG: %[[a1:.+]] = extractelement <4 x i32> %a, i64 1
  ; CHECK-DAG: %[[a2:.+]] = extractelement <4 x i32> %a, i64 2
  ; CHECK-DAG: %[[a3:.+]] = extractelement <4 x i32> %a, i64 3
  ; CHECK-DAG: %[[b0:.+]] = extractelement <4 x i32> %b, i64 0
  ; CHECK-DAG: %[[b1:.+]] = extractelement <4 x i32> %b, i64 1
  ; CHECK-DAG: %[[b2:.+]] = extractelement <4 x i32> %b, i64 2
  ; CHECK-DAG: %[[b3:.+]] = extractelement <4 x i32> %b, i64 3
  ; CHECK-DAG: %[[c0:.+]] = extractelement <4 x i32> %c, i64 0
  ; CHECK-DAG: %[[c1:.+]] = extractelement <4 x i32> %c, i64 1
  ; CHECK-DAG: %[[c2:.+]] = extractelement <4 x i32> %c, i64 2
  ; CHECK-DAG: %[[c3:.+]] = extractelement <4 x i32> %c, i64 3
  ; CHECK-DAG: %[[max0:.+]] = call i32 @dx.op.binary.i32(i32 37, i32 %[[a0]], i32 %[[b0]])
  ; CHECK-DAG: %[[max1:.+]] = call i32 @dx.op.binary.i32(i32 37, i32 %[[a1]], i32 %[[b1]])
  ; CHECK-DAG: %[[max2:.+]] = call i32 @dx.op.binary.i32(i32 37, i32 %[[a2]], i32 %[[b2]])
  ; CHECK-DAG: %[[max3:.+]] = call i32 @dx.op.binary.i32(i32 37, i32 %[[a3]], i32 %[[b3]])
  ; CHECK-DAG: %[[min0:.+]] = call i32 @dx.op.binary.i32(i32 38, i32 %[[max0]], i32 %[[c0]])
  ; CHECK-DAG: %[[min1:.+]] = call i32 @dx.op.binary.i32(i32 38, i32 %[[max1]], i32 %[[c1]])
  ; CHECK-DAG: %[[min2:.+]] = call i32 @dx.op.binary.i32(i32 38, i32 %[[max2]], i32 %[[c2]])
  ; CHECK-DAG: %[[min3:.+]] = call i32 @dx.op.binary.i32(i32 38, i32 %[[max3]], i32 %[[c3]])
  ; CHECK-DAG: %[[ret0:.+]] = insertelement <4 x i32> poison, i32 %[[min0]], i64 0
  ; CHECK-DAG: %[[ret1:.+]] = insertelement <4 x i32> %[[ret0]], i32 %[[min1]], i64 1
  ; CHECK-DAG: %[[ret2:.+]] = insertelement <4 x i32> %[[ret1]], i32 %[[min2]], i64 2
  ; CHECK-DAG: %[[ret3:.+]] = insertelement <4 x i32> %[[ret2]], i32 %[[min3]], i64 3
  ; CHECK: ret <4 x i32> %[[ret3]]
  %dx.clamp = call <4 x i32> @llvm.dx.sclamp.v4i32(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c)
  ret <4 x i32> %dx.clamp
}

; CHECK-LABEL: clamp_uint16_t3
define noundef <3 x i16> @clamp_uint16_t3(<3 x i16> noundef %a, <3 x i16> noundef %b, <3 x i16> noundef %c) {
entry:
  ; CHECK-DAG: %[[a0:.+]] = extractelement <3 x i16> %a, i64 0
  ; CHECK-DAG: %[[a1:.+]] = extractelement <3 x i16> %a, i64 1
  ; CHECK-DAG: %[[a2:.+]] = extractelement <3 x i16> %a, i64 2
  ; CHECK-DAG: %[[b0:.+]] = extractelement <3 x i16> %b, i64 0
  ; CHECK-DAG: %[[b1:.+]] = extractelement <3 x i16> %b, i64 1
  ; CHECK-DAG: %[[b2:.+]] = extractelement <3 x i16> %b, i64 2
  ; CHECK-DAG: %[[c0:.+]] = extractelement <3 x i16> %c, i64 0
  ; CHECK-DAG: %[[c1:.+]] = extractelement <3 x i16> %c, i64 1
  ; CHECK-DAG: %[[c2:.+]] = extractelement <3 x i16> %c, i64 2
  ; CHECK-DAG: %[[max0:.+]] = call i16 @dx.op.binary.i16(i32 39, i16 %[[a0]], i16 %[[b0]])
  ; CHECK-DAG: %[[max1:.+]] = call i16 @dx.op.binary.i16(i32 39, i16 %[[a1]], i16 %[[b1]])
  ; CHECK-DAG: %[[max2:.+]] = call i16 @dx.op.binary.i16(i32 39, i16 %[[a2]], i16 %[[b2]])
  ; CHECK-DAG: %[[min0:.+]] = call i16 @dx.op.binary.i16(i32 40, i16 %[[max0]], i16 %[[c0]])
  ; CHECK-DAG: %[[min1:.+]] = call i16 @dx.op.binary.i16(i32 40, i16 %[[max1]], i16 %[[c1]])
  ; CHECK-DAG: %[[min2:.+]] = call i16 @dx.op.binary.i16(i32 40, i16 %[[max2]], i16 %[[c2]])
  ; CHECK-DAG: %[[ret0:.+]] = insertelement <3 x i16> poison, i16 %[[min0]], i64 0
  ; CHECK-DAG: %[[ret1:.+]] = insertelement <3 x i16> %[[ret0]], i16 %[[min1]], i64 1
  ; CHECK-DAG: %[[ret2:.+]] = insertelement <3 x i16> %[[ret1]], i16 %[[min2]], i64 2
  ; CHECK: ret <3 x i16> %[[ret2]]
  %dx.clamp = call <3 x i16> @llvm.dx.uclamp.v3i16(<3 x i16> %a, <3 x i16> %b, <3 x i16> %c)
  ret <3 x i16> %dx.clamp
}

; CHECK-LABEL: clamp_uint4
define noundef <4 x i32> @clamp_uint4(<4 x i32> noundef %a, <4 x i32> noundef %b, <4 x i32> noundef %c) {
entry:
  ; CHECK-DAG: %[[a0:.+]] = extractelement <4 x i32> %a, i64 0
  ; CHECK-DAG: %[[a1:.+]] = extractelement <4 x i32> %a, i64 1
  ; CHECK-DAG: %[[a2:.+]] = extractelement <4 x i32> %a, i64 2
  ; CHECK-DAG: %[[a3:.+]] = extractelement <4 x i32> %a, i64 3
  ; CHECK-DAG: %[[b0:.+]] = extractelement <4 x i32> %b, i64 0
  ; CHECK-DAG: %[[b1:.+]] = extractelement <4 x i32> %b, i64 1
  ; CHECK-DAG: %[[b2:.+]] = extractelement <4 x i32> %b, i64 2
  ; CHECK-DAG: %[[b3:.+]] = extractelement <4 x i32> %b, i64 3
  ; CHECK-DAG: %[[c0:.+]] = extractelement <4 x i32> %c, i64 0
  ; CHECK-DAG: %[[c1:.+]] = extractelement <4 x i32> %c, i64 1
  ; CHECK-DAG: %[[c2:.+]] = extractelement <4 x i32> %c, i64 2
  ; CHECK-DAG: %[[c3:.+]] = extractelement <4 x i32> %c, i64 3
  ; CHECK-DAG: %[[max0:.+]] = call i32 @dx.op.binary.i32(i32 39, i32 %[[a0]], i32 %[[b0]])
  ; CHECK-DAG: %[[max1:.+]] = call i32 @dx.op.binary.i32(i32 39, i32 %[[a1]], i32 %[[b1]])
  ; CHECK-DAG: %[[max2:.+]] = call i32 @dx.op.binary.i32(i32 39, i32 %[[a2]], i32 %[[b2]])
  ; CHECK-DAG: %[[max3:.+]] = call i32 @dx.op.binary.i32(i32 39, i32 %[[a3]], i32 %[[b3]])
  ; CHECK-DAG: %[[min0:.+]] = call i32 @dx.op.binary.i32(i32 40, i32 %[[max0]], i32 %[[c0]])
  ; CHECK-DAG: %[[min1:.+]] = call i32 @dx.op.binary.i32(i32 40, i32 %[[max1]], i32 %[[c1]])
  ; CHECK-DAG: %[[min2:.+]] = call i32 @dx.op.binary.i32(i32 40, i32 %[[max2]], i32 %[[c2]])
  ; CHECK-DAG: %[[min3:.+]] = call i32 @dx.op.binary.i32(i32 40, i32 %[[max3]], i32 %[[c3]])
  ; CHECK-DAG: %[[ret0:.+]] = insertelement <4 x i32> poison, i32 %[[min0]], i64 0
  ; CHECK-DAG: %[[ret1:.+]] = insertelement <4 x i32> %[[ret0]], i32 %[[min1]], i64 1
  ; CHECK-DAG: %[[ret2:.+]] = insertelement <4 x i32> %[[ret1]], i32 %[[min2]], i64 2
  ; CHECK-DAG: %[[ret3:.+]] = insertelement <4 x i32> %[[ret2]], i32 %[[min3]], i64 3
  ; CHECK: ret <4 x i32> %[[ret3]]
  %dx.clamp = call <4 x i32> @llvm.dx.uclamp.v4i32(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c)
  ret <4 x i32> %dx.clamp
}

; CHECK-LABEL: clamp_uint64_t4
define noundef <2 x i64> @clamp_uint64_t4(<2 x i64> noundef %a, <2 x i64> noundef %b, <2 x i64> noundef %c) {
entry:
  ; CHECK-DAG: %[[a0:.+]] = extractelement <2 x i64> %a, i64 0
  ; CHECK-DAG: %[[a1:.+]] = extractelement <2 x i64> %a, i64 1
  ; CHECK-DAG: %[[b0:.+]] = extractelement <2 x i64> %b, i64 0
  ; CHECK-DAG: %[[b1:.+]] = extractelement <2 x i64> %b, i64 1
  ; CHECK-DAG: %[[c0:.+]] = extractelement <2 x i64> %c, i64 0
  ; CHECK-DAG: %[[c1:.+]] = extractelement <2 x i64> %c, i64 1
  ; CHECK-DAG: %[[max0:.+]] = call i64 @dx.op.binary.i64(i32 39, i64 %[[a0]], i64 %[[b0]])
  ; CHECK-DAG: %[[max1:.+]] = call i64 @dx.op.binary.i64(i32 39, i64 %[[a1]], i64 %[[b1]])
  ; CHECK-DAG: %[[min0:.+]] = call i64 @dx.op.binary.i64(i32 40, i64 %[[max0]], i64 %[[c0]])
  ; CHECK-DAG: %[[min1:.+]] = call i64 @dx.op.binary.i64(i32 40, i64 %[[max1]], i64 %[[c1]])
  ; CHECK-DAG: %[[ret0:.+]] = insertelement <2 x i64> poison, i64 %[[min0]], i64 0
  ; CHECK-DAG: %[[ret1:.+]] = insertelement <2 x i64> %[[ret0]], i64 %[[min1]], i64 1
  ; CHECK: ret <2 x i64> %[[ret1]]
  %dx.clamp = call <2 x i64> @llvm.dx.uclamp.v2i64(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c)
  ret <2 x i64> %dx.clamp
}


declare <3 x half> @llvm.dx.nclamp.v3f16(<3 x half>, <3 x half>, <3 x half>)
declare <4 x float> @llvm.dx.nclamp.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <2 x double> @llvm.dx.nclamp.v2f64(<2 x double>, <2 x double>, <2 x double>)
declare <4 x i32> @llvm.dx.sclamp.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
declare <3 x i16> @llvm.dx.uclamp.v3i32(<3 x i16>, <3 x i32>, <3 x i16>)
declare <4 x i32> @llvm.dx.uclamp.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
declare <2 x i64> @llvm.dx.uclamp.v2i64(<2 x i64>, <2 x i64>, <2 x i64>)

