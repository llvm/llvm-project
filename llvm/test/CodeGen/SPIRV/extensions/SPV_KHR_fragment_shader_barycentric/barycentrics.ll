; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-pixel -spirv-ext=+SPV_KHR_fragment_shader_barycentric %s -o - | FileCheck %s --implicit-check-not='OpCapability FragmentBarycentricNV' --implicit-check-not='BuiltIn BaryCoordNV' --implicit-check-not='BuiltIn BaryCoordNoPerspNV'
; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel -spirv-ext=-SPV_KHR_fragment_shader_barycentric %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel -spirv-ext=+SPV_KHR_fragment_shader_barycentric %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Exercise the decorated input loads emitted for SV_Barycentrics, including both
; perspective modes and centroid/sample interpolation in one fragment shader.
; ERROR: LLVM ERROR: Barycentric built-ins require the following SPIR-V extension: SPV_KHR_fragment_shader_barycentric

; CHECK-DAG: OpCapability FragmentBarycentricKHR
; CHECK-DAG: OpCapability SampleRateShading
; CHECK-DAG: OpExtension "SPV_KHR_fragment_shader_barycentric"
; CHECK-DAG: OpDecorate %[[#BARY:]] BuiltIn BaryCoordKHR
; CHECK-DAG: OpDecorate %[[#BARY]] Centroid
; CHECK-DAG: OpDecorate %[[#NOPERSP:]] BuiltIn BaryCoordNoPerspKHR
; CHECK-DAG: OpDecorate %[[#NOPERSP]] Sample
; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FLOAT3:]] = OpTypeVector %[[#FLOAT]] 3
; CHECK-DAG: %[[#INPUT:]] = OpTypePointer Input %[[#FLOAT3]]
; CHECK-DAG: %[[#BARY]] = OpVariable %[[#INPUT]] Input
; CHECK-DAG: %[[#NOPERSP]] = OpVariable %[[#INPUT]] Input

@SV_Barycentrics = external hidden thread_local addrspace(7) externally_initialized constant <3 x float>, !spirv.Decorations !0
@SV_BarycentricsNoPersp = external hidden thread_local addrspace(7) externally_initialized constant <3 x float>, !spirv.Decorations !2
@SV_Target = external hidden thread_local addrspace(8) global <3 x float>, !spirv.Decorations !4

define void @main() #0 {
; CHECK: OpLoad %[[#FLOAT3]] %[[#BARY]]
; CHECK: OpLoad %[[#FLOAT3]] %[[#NOPERSP]]
  %perspective = load <3 x float>, ptr addrspace(7) @SV_Barycentrics, align 16
  %noperspective = load <3 x float>, ptr addrspace(7) @SV_BarycentricsNoPersp, align 16
  %sum = fadd <3 x float> %perspective, %noperspective
  store <3 x float> %sum, ptr addrspace(8) @SV_Target, align 16
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!0 = !{!1, !6}
!1 = !{i32 11, i32 5286}
!2 = !{!3, !7}
!3 = !{i32 11, i32 5287}
!4 = !{!5}
!5 = !{i32 30, i32 0}
!6 = !{i32 16} ; Centroid
!7 = !{i32 17} ; Sample
