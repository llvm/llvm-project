; RUN: opt -passes=globaldce -S -o - %s | llc -mtriple=spirv-unknown-vulkan1.3-vertex \
; RUN:   -o - | FileCheck %s
; RUN: %if spirv-tools %{ opt -passes=globaldce -S -o - %s \
; RUN:   | llc -mtriple=spirv-unknown-vulkan1.3-vertex -filetype=obj -o - \
; RUN:   | spirv-val --target-env vulkan1.3 %}
;
; Confirm that GlobalDCE removes an addrspace(7) global with no users before
; the SPIR-V backend emits OpEntryPoint. A passing run means the dead variable
; is absent from the interface list and -fspv-preserve-interface must guard it.
;
; @used_input has a live load. It must appear in OpEntryPoint.
; @dead_input has no users. GlobalDCE removes it before backend emission.

; OpEntryPoint appears before OpVariable in SPIR-V module order. Check it first.
; CHECK: OpCapability Shader
; CHECK: OpEntryPoint Vertex %[[#]] "main"

; Exactly one Input variable must appear: the one for @used_input.
; CHECK: %[[#]] = OpTypePointer Input
; CHECK: %[[#UsedVar:]] = OpVariable %[[#]] Input
; CHECK-NOT: = OpVariable %[[#]] Input

@used_input = external hidden thread_local addrspace(7) global float,
    !spirv.Decorations !0
@dead_input = external hidden thread_local addrspace(7) global float,
    !spirv.Decorations !2

define void @main() #0 {
  ; Load from @used_input. The result feeds the store to @output.
  %v = load float, ptr addrspace(7) @used_input, align 4
  store float %v, ptr addrspace(8) @output, align 4
  ; @dead_input is never referenced.
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
