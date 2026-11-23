; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-vertex %s -o - | FileCheck %s --implicit-check-not=OpExtension
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.2-vertex %s -o - | FileCheck %s --implicit-check-not=OpExtension
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-vertex %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Before SPIR-V 1.3, DrawParameters requires SPV_KHR_shader_draw_parameters.
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32v1.2-unknown-vulkan1.0-vertex -spirv-ext=+SPV_KHR_shader_draw_parameters %s -o - | FileCheck %s --check-prefixes=CHECK,EXT
; RUN: not llc -O0 -mtriple=spirv32v1.2-unknown-vulkan1.0-vertex %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not llc -O0 -mtriple=spirv32v1.2-unknown-vulkan1.0-vertex -spirv-ext=-SPV_KHR_shader_draw_parameters %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR

; SV_StartInstanceLocation is lowered to a load of the BaseInstance built-in, which
; requires DrawParameters. The capability is core from SPIR-V 1.3, so no
; extension is needed for Vulkan 1.2 or later.

; ERROR: LLVM ERROR: capability DrawParameters requires SPIR-V 1.3 or the following SPIR-V extension: SPV_KHR_shader_draw_parameters

; CHECK-DAG: OpCapability DrawParameters
; EXT-DAG: OpExtension "SPV_KHR_shader_draw_parameters"
; CHECK-DAG: OpDecorate %[[#START:]] BuiltIn BaseInstance
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer Input %[[#INT]]
; CHECK-DAG: %[[#START]] = OpVariable %[[#PTR]] Input

@SV_StartInstanceLocation = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations !0
@SV_Position = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations !2

define void @main() #0 {
entry:
; CHECK: OpLoad %[[#INT]] %[[#START]]
  %0 = load i32, ptr addrspace(7) @SV_StartInstanceLocation, align 4
  %1 = uitofp i32 %0 to float
  %2 = insertelement <4 x float> zeroinitializer, float %1, i32 0
  store <4 x float> %2, ptr addrspace(8) @SV_Position, align 16
  ret void
}

attributes #0 = { "hlsl.shader"="vertex" }

!0 = !{!1}
!1 = !{i32 11, i32 4425}
!2 = !{!3}
!3 = !{i32 11, i32 0}
