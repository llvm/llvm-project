; RUN: llc -mtriple=spirv-unknown-vulkan1.3-vertex -o - %s | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv-unknown-vulkan1.3-vertex \
; RUN:   -filetype=obj -o - %s | spirv-val --target-env vulkan1.3 %}
;
; Confirm that addrspace(7) globals appear in the SPIR-V output as distinct
; OpVariables and in the OpEntryPoint interface. Three cases are tested:
;
;   @used_input   -- has a real load in @main.
;   @dead_input   -- no IR uses; protected by llvm.compiler.used.
;   @bare_input   -- no IR uses; NOT in llvm.compiler.used.
;
; @bare_input tests that the backend emits any addrspace(7) global that reaches
; it without an initializer and without real function uses, regardless of
; llvm.compiler.used.
;
; The OpEntryPoint check pins the interface to exactly four IDs. The SPIR-V
; backend builds the interface by iterating every Input/Output OpVariable in
; the module, so combined with the four OpVariable checks below this proves
; all four preserved variables appear in OpEntryPoint regardless of the
; backend's interface ordering.

; CHECK: OpCapability Shader
; CHECK: OpEntryPoint Vertex %[[#]] "main" %[[#]] %[[#]] %[[#]] %[[#]]

; CHECK-DAG: OpDecorate %[[#USED:]] Location 0
; CHECK-DAG: OpDecorate %[[#DEAD:]] Location 1
; CHECK-DAG: OpDecorate %[[#BARE:]] Location 2
; CHECK-DAG: OpDecorate %[[#OUTPUT:]] Location 0
; CHECK-DAG: %[[#USED]] = OpVariable %[[#]] Input
; CHECK-DAG: %[[#DEAD]] = OpVariable %[[#]] Input
; CHECK-DAG: %[[#BARE]] = OpVariable %[[#]] Input
; CHECK-DAG: %[[#OUTPUT]] = OpVariable %[[#]] Output

@used_input = external hidden thread_local addrspace(7) global float,
    !spirv.Decorations !0
@dead_input = external hidden thread_local addrspace(7) global float,
    !spirv.Decorations !2
@bare_input = external hidden thread_local addrspace(7) global float,
    !spirv.Decorations !4

@llvm.compiler.used = appending global [1 x ptr addrspace(7)]
    [ptr addrspace(7) @dead_input], section "llvm.metadata"

define void @main() #0 {
  %v = load float, ptr addrspace(7) @used_input, align 4
  store float %v, ptr addrspace(8) @output, align 4
  ret void
}

@output = external hidden thread_local addrspace(8) global float,
    !spirv.Decorations !6

attributes #0 = { "hlsl.shader"="vertex" }

; Location = 0 on @used_input
!0 = !{!1}
!1 = !{i32 30, i32 0}

; Location = 1 on @dead_input
!2 = !{!3}
!3 = !{i32 30, i32 1}

; Location = 2 on @bare_input
!4 = !{!5}
!5 = !{i32 30, i32 2}

; Location = 0 on @output
!6 = !{!7}
!7 = !{i32 30, i32 0}
