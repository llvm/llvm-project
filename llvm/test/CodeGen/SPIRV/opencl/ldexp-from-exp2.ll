; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; The fold is not environment gated. The match is covered by the Vulkan test,
; this checks the OpenCL.std mapping.

; CHECK-DAG: %[[#ExtInstSetId:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32

; CHECK-LABEL: Begin function test_i32_exponent{{$}}
; CHECK-NOT: exp2
; CHECK: %[[#]] = OpExtInst %[[#Float]] %[[#ExtInstSetId]] ldexp
define spir_kernel void @test_i32_exponent(float %x, i32 %k, ptr %out) {
  %e = sitofp i32 %k to float
  %t = call reassoc float @llvm.exp2.f32(float %e)
  %r = fmul reassoc float %t, %x
  store float %r, ptr %out
  ret void
}

declare float @llvm.exp2.f32(float)
