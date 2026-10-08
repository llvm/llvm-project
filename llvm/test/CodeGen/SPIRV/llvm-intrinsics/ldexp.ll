; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#ExtInstSetId:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#Half:]] = OpTypeFloat 16
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Double:]] = OpTypeFloat 64
; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Float4:]] = OpTypeVector %[[#Float]] 4
; CHECK-DAG: %[[#Int4:]] = OpTypeVector %[[#Int32]] 4
; CHECK-DAG: %[[#Float8:]] = OpTypeVector %[[#Float]] 8
; CHECK-DAG: %[[#Int8:]] = OpTypeVector %[[#Int32]] 8

define spir_func void @test_ldexp(ptr %xh, ptr %xf, ptr %xd, ptr %xv,
                                  half %h, float %f, double %d, <4 x float> %vf,
                                  i32 %k) {
entry:
  %0 = call half @llvm.ldexp.f16.i32(half %h, i32 %k)
  store half %0, ptr %xh
  %1 = call float @llvm.ldexp.f32.i32(float %f, i32 %k)
  store float %1, ptr %xf
  %2 = call double @llvm.ldexp.f64.i32(double %d, i32 %k)
  store double %2, ptr %xd
  %3 = call <4 x float> @llvm.ldexp.v4f32.i32(<4 x float> %vf, i32 %k)
  store <4 x float> %3, ptr %xv
; CHECK: %[[#]] = OpExtInst %[[#Half]] %[[#ExtInstSetId]] ldexp
; CHECK: %[[#]] = OpExtInst %[[#Float]] %[[#ExtInstSetId]] ldexp
; CHECK: %[[#]] = OpExtInst %[[#Double]] %[[#ExtInstSetId]] ldexp
; CHECK: %[[#Splat:]] = OpCompositeConstruct %[[#Int4]]
; CHECK: %[[#]] = OpExtInst %[[#Float4]] %[[#ExtInstSetId]] ldexp %[[#]] %[[#Splat]]
  ret void
}

define spir_func void @test_ldexp_v8(ptr %xv, <8 x float> %vf, i32 %k) {
entry:
  %0 = call <8 x float> @llvm.ldexp.v8f32.i32(<8 x float> %vf, i32 %k)
  store <8 x float> %0, ptr %xv
; CHECK: %[[#Splat8:]] = OpCompositeConstruct %[[#Int8]]
; CHECK: %[[#]] = OpExtInst %[[#Float8]] %[[#ExtInstSetId]] ldexp %[[#]] %[[#Splat8]]
  ret void
}

declare half @llvm.ldexp.f16.i32(half, i32)
declare float @llvm.ldexp.f32.i32(float, i32)
declare double @llvm.ldexp.f64.i32(double, i32)
declare <4 x float> @llvm.ldexp.v4f32.i32(<4 x float>, i32)
declare <8 x float> @llvm.ldexp.v8f32.i32(<8 x float>, i32)
