; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-pixel -spirv-ext=+SPV_KHR_fragment_shading_rate %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel -spirv-ext=+SPV_KHR_fragment_shading_rate %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Simplified LLVM IR of a pixel shader reading SV_ShadingRate.

; CHECK-DAG: OpCapability FragmentShadingRateKHR
; CHECK-DAG: OpExtension "SPV_KHR_fragment_shading_rate"
; CHECK-DAG: OpDecorate %[[#RATE:]] BuiltIn ShadingRateKHR
; CHECK-DAG: OpDecorate %[[#RATE]] Flat
; CHECK-DAG: OpDecorate %[[#TARGET:]] Location 0

; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ptr_i:]] = OpTypePointer Input %[[#int]]
; CHECK-DAG: %[[#RATE]] = OpVariable %[[#ptr_i]] Input

@SV_ShadingRate = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations !0
@SV_Target = external hidden thread_local addrspace(8) global float, !spirv.Decorations !3

define void @main() #0 {
entry:
  %0 = load i32, ptr addrspace(7) @SV_ShadingRate, align 4
  %conv = uitofp i32 %0 to float
  store float %conv, ptr addrspace(8) @SV_Target, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

; BuiltIn ShadingRateKHR (4444) and Flat (14) in a single node.
!0 = !{!1, !2}
!1 = !{i32 11, i32 4444}
!2 = !{i32 14}
; SV_Target output: Location 0.
!3 = !{!4}
!4 = !{i32 30, i32 0}
