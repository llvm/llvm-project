; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

@v1 = addrspace(1) global i32 42, !spirv.Decorations !2
@v2 = addrspace(1) global float 1.0, !spirv.Decorations !4

define i32 @use_globals() {
entry:
  %v1_ptr = addrspacecast i32 addrspace(1)* @v1 to i32*
  %v1_val = load i32, i32* %v1_ptr

  %v2_ptr = addrspacecast float addrspace(1)* @v2 to float*
  %v2_val = load float, float* %v2_ptr
  %v2_int = fptosi float %v2_val to i32

  %sum = add i32 %v1_val, %v2_int
  ret i32 %sum
}

; CHECK: OpDecorate %[[#PId1:]] Constant
; CHECK: OpDecorate %[[#PId2:]] Constant
; CHECK: OpDecorate %[[#PId2]] Binding 1
; CHECK: %[[#PId1]] = OpVariable %[[#]] CrossWorkgroup %[[#]]
; CHECK: %[[#PId2]] = OpVariable %[[#]] CrossWorkgroup %[[#]]

!1 = !{i32 22}                          ; Constant
!2 = !{!1}                             ; @v1
!3 = !{i32 33, i32 1}                  ; Binding 1
!4 = !{!1, !3}                         ; @v2
