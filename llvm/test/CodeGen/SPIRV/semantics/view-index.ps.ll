; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - | FileCheck %s --implicit-check-not=OpExtension
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.2-pixel %s -o - | FileCheck %s --implicit-check-not=OpExtension
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Before SPIR-V 1.3, MultiView requires SPV_KHR_multiview.
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32v1.2-unknown-vulkan1.0-pixel -spirv-ext=+SPV_KHR_multiview %s -o - | FileCheck %s --check-prefixes=CHECK,EXT
; RUN: not llc -O0 -mtriple=spirv32v1.2-unknown-vulkan1.0-pixel %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not llc -O0 -mtriple=spirv32v1.2-unknown-vulkan1.0-pixel -spirv-ext=-SPV_KHR_multiview %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR

; SV_ViewID is lowered to a flat load of the ViewIndex built-in, which requires
; MultiView. The capability is core from SPIR-V 1.3, so no extension is needed
; for Vulkan 1.2 or later.

; ERROR: LLVM ERROR: capability MultiView requires SPIR-V 1.3 or the following SPIR-V extension: SPV_KHR_multiview

; CHECK-DAG: OpCapability MultiView
; EXT-DAG: OpExtension "SPV_KHR_multiview"
; CHECK-DAG: OpDecorate %[[#VIEW:]] BuiltIn ViewIndex
; CHECK-DAG: OpDecorate %[[#VIEW]] Flat
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer Input %[[#INT]]
; CHECK-DAG: %[[#VIEW]] = OpVariable %[[#PTR]] Input

@SV_ViewID = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations !0
@SV_Target0 = external hidden thread_local addrspace(8) global float, !spirv.Decorations !3

define void @main() #0 {
entry:
; CHECK: OpLoad %[[#INT]] %[[#VIEW]]
  %0 = load i32, ptr addrspace(7) @SV_ViewID, align 4
  %1 = uitofp i32 %0 to float
  store float %1, ptr addrspace(8) @SV_Target0, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!0 = !{!1, !2}
!1 = !{i32 11, i32 4440}
!2 = !{i32 14}
!3 = !{!4}
!4 = !{i32 30, i32 0}
