; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#ExtInstSetId:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#Half:]] = OpTypeFloat 16
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Double:]] = OpTypeFloat 64
; CHECK-DAG: %[[#Float4:]] = OpTypeVector %[[#Float]] 4
; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Int4:]] = OpTypeVector %[[#Int32]] 4

; CHECK-LABEL: Begin function test_ldexp_f16
; CHECK: %[[#]] = OpExtInst %[[#Half]] %[[#ExtInstSetId]] Ldexp
define half @test_ldexp_f16(half %x, i32 %k) {
  %res = call half @llvm.ldexp.f16.i32(half %x, i32 %k)
  ret half %res
}

; CHECK-LABEL: Begin function test_ldexp_f32
; CHECK: %[[#]] = OpExtInst %[[#Float]] %[[#ExtInstSetId]] Ldexp
define float @test_ldexp_f32(float %x, i32 %k) {
  %res = call float @llvm.ldexp.f32.i32(float %x, i32 %k)
  ret float %res
}

; CHECK-LABEL: Begin function test_ldexp_f64
; CHECK: %[[#]] = OpExtInst %[[#Double]] %[[#ExtInstSetId]] Ldexp
define double @test_ldexp_f64(double %x, i32 %k) {
  %res = call double @llvm.ldexp.f64.i32(double %x, i32 %k)
  ret double %res
}

; CHECK-LABEL: Begin function test_ldexp_v4f32
; CHECK: %[[#Splat:]] = OpCompositeConstruct %[[#Int4]]
; CHECK: %[[#]] = OpExtInst %[[#Float4]] %[[#ExtInstSetId]] Ldexp %[[#]] %[[#Splat]]
define <4 x float> @test_ldexp_v4f32(<4 x float> %x, i32 %k) {
  %res = call <4 x float> @llvm.ldexp.v4f32.i32(<4 x float> %x, i32 %k)
  ret <4 x float> %res
}

declare half @llvm.ldexp.f16.i32(half, i32)
declare float @llvm.ldexp.f32.i32(float, i32)
declare double @llvm.ldexp.f64.i32(double, i32)
declare <4 x float> @llvm.ldexp.v4f32.i32(<4 x float>, i32)
