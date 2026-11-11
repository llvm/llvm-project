; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - | FileCheck %s --implicit-check-not=OpExtension
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; SV_SampleIndex is lowered to a flat load of the SampleId built-in, which
; requires SampleRateShading.

; CHECK-DAG: OpCapability SampleRateShading
; CHECK-DAG: OpDecorate %[[#SAMPLE:]] BuiltIn SampleId
; CHECK-DAG: OpDecorate %[[#SAMPLE]] Flat
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer Input %[[#INT]]
; CHECK-DAG: %[[#SAMPLE]] = OpVariable %[[#PTR]] Input

@SV_SampleIndex = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations !0
@SV_Target0 = external hidden thread_local addrspace(8) global float, !spirv.Decorations !3

define void @main() #0 {
entry:
; CHECK: OpLoad %[[#INT]] %[[#SAMPLE]]
  %0 = load i32, ptr addrspace(7) @SV_SampleIndex, align 4
  %1 = uitofp i32 %0 to float
  store float %1, ptr addrspace(8) @SV_Target0, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!0 = !{!1, !2}
!1 = !{i32 11, i32 18}
!2 = !{i32 14}
!3 = !{!4}
!4 = !{i32 30, i32 0}
