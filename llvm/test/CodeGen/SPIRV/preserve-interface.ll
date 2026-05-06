; RUN: llc -mtriple=spirv-unknown-vulkan1.3-vertex -o - %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv-unknown-vulkan1.3-vertex \
; RUN:   -filetype=obj -o - %s | spirv-val --target-env vulkan1.3 %}
;
; Confirm that an addrspace(7) global protected by llvm.compiler.used appears
; in the SPIR-V output as a distinct OpVariable, even though it has no IR users.
;
; @used_input has a real load. @dead_input is only in llvm.compiler.used.
;
; %[[#USED]] and %[[#DEAD]] capture different IDs (Location 0 vs 1), so the
; OpVariable checks below require two distinct OpVariable Input instructions.
;
; Without the processGlobalValue fix in SPIRVEmitIntrinsics.cpp, @dead_input
; gets no spv_unref_global, buildGlobalVariable is never called for it, and
; both OpDecorate Location 1 and the second OpVariable Input are absent.

; CHECK: OpCapability Shader
; CHECK: OpEntryPoint Vertex %[[#]] "main"

; CHECK-DAG: OpDecorate %[[#USED:]] Location 0
; CHECK-DAG: OpDecorate %[[#DEAD:]] Location 1
; CHECK-DAG: %[[#USED]] = OpVariable %[[#]] Input
; CHECK-DAG: %[[#DEAD]] = OpVariable %[[#]] Input
; CHECK-DAG: %[[#]]     = OpVariable %[[#]] Output

@used_input = external hidden thread_local addrspace(7) global float,
    !spirv.Decorations !0
@dead_input = external hidden thread_local addrspace(7) global float,
    !spirv.Decorations !2

@llvm.compiler.used = appending global [1 x ptr addrspace(7)]
    [ptr addrspace(7) @dead_input], section "llvm.metadata"

define void @main() #0 {
  %v = load float, ptr addrspace(7) @used_input, align 4
  store float %v, ptr addrspace(8) @output, align 4
  ret void
}

@output = external hidden thread_local addrspace(8) global float,
    !spirv.Decorations !4

attributes #0 = { "hlsl.shader"="vertex" }

; Location = 0 on @used_input
!0 = !{!1}
!1 = !{i32 30, i32 0}

; Location = 1 on @dead_input
!2 = !{!3}
!3 = !{i32 30, i32 1}

; Location = 0 on @output
!4 = !{!5}
!5 = !{i32 30, i32 0}
