; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-pixel -spirv-ext=+SPV_KHR_fragment_shader_barycentric %s -o - | FileCheck %s --implicit-check-not='BuiltIn BaryCoordNV' --implicit-check-not='BuiltIn BaryCoordNoPerspNV'
; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel -spirv-ext=+SPV_KHR_fragment_shader_barycentric %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; The NV built-in names are aliases of the KHR built-ins, so they are still
; recognized and get the same decorations and requirements.
; CHECK-ERROR: LLVM ERROR: Barycentric built-ins require the following SPIR-V extension: SPV_KHR_fragment_shader_barycentric
; CHECK-DAG: OpCapability FragmentBarycentricKHR
; CHECK-DAG: OpExtension "SPV_KHR_fragment_shader_barycentric"
; CHECK-DAG: OpDecorate %[[#NOPERSP:]] BuiltIn BaryCoordNoPerspKHR
; CHECK-DAG: OpDecorate %[[#BARY:]] BuiltIn BaryCoordKHR
; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FLOAT3:]] = OpTypeVector %[[#FLOAT]] 3
; CHECK-DAG: %[[#INPUT:]] = OpTypePointer Input %[[#FLOAT3]]
; CHECK-DAG: %[[#NOPERSP]] = OpVariable %[[#INPUT]] Input
; CHECK-DAG: %[[#BARY]] = OpVariable %[[#INPUT]] Input

@__spirv_BuiltInBaryCoordNoPerspNV = external hidden addrspace(7) constant <3 x float>
@__spirv_BuiltInBaryCoordNV = external hidden addrspace(7) constant <3 x float>
@target = external hidden addrspace(8) global <3 x float>, !spirv.Decorations !0

define void @main() #0 {
; CHECK: OpLoad %[[#FLOAT3]] %[[#NOPERSP]]
; CHECK: OpLoad %[[#FLOAT3]] %[[#BARY]]
  %noperspective = load <3 x float>, ptr addrspace(7) @__spirv_BuiltInBaryCoordNoPerspNV, align 16
  %perspective = load <3 x float>, ptr addrspace(7) @__spirv_BuiltInBaryCoordNV, align 16
  %sum = fadd <3 x float> %noperspective, %perspective
  store <3 x float> %sum, ptr addrspace(8) @target, align 16
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!0 = !{!1}
!1 = !{i32 30, i32 0}
