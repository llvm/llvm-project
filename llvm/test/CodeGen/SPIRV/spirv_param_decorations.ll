; Test that !spirv.ParameterDecorations metadata is correctly translated
; into OpDecorate instructions on function parameters.

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define spir_kernel void @k(ptr addrspace(1) %a, float %b, ptr addrspace(1) %c) !spirv.ParameterDecorations !14 {
entry:
  ret void
}

; CHECK-SPIRV: OpDecorate %[[#PId1:]] Restrict
; CHECK-SPIRV: OpDecorate %[[#PId1]] Alignment 4
; CHECK-SPIRV: OpDecorate %[[#PId2:]] Volatile
; CHECK-SPIRV: %[[#PId1]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV: %[[#PId2]] = OpFunctionParameter %[[#]]

!8 = !{i32 19}
!9 = !{i32 44, i32 4}
!10 = !{i32 21}
!11 = !{!8, !9}
!12 = !{}
!13 = !{!10}
!14 = !{!11, !12, !13}
