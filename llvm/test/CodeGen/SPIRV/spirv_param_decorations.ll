; Test that !spirv.ParameterDecorations metadata is correctly translated
; into OpDecorate instructions on function parameters.

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv32v1.2-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: not --crash llc -O0 -mtriple=spirv32v1.1-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-V11

; CHECK-V11: LLVM ERROR: Adding SPIR-V requirements this target can't satisfy.

define spir_kernel void @k(ptr addrspace(1) %a, float %b, ptr addrspace(1) %c) !spirv.ParameterDecorations !14 {
entry:
  ret void
}

; CHECK-SPIRV: OpDecorate %[[#PId1:]] Restrict
; CHECK-SPIRV: OpDecorate %[[#PId1]] Alignment 4
; CHECK-SPIRV: OpDecorate %[[#PId2:]] Volatile
; CHECK-SPIRV: OpDecorateId %[[#PId2]] AlignmentId %[[#C16:]]
; CHECK-SPIRV: OpDecorateId %[[#PId2]] MaxByteOffsetId %[[#C0:]]
; CHECK-SPIRV-DAG: %[[#C16]] = OpConstant %[[#]] 16
; CHECK-SPIRV-DAG: %[[#C0]] = OpConstantNull %[[#]]
; CHECK-SPIRV: %[[#PId1]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV: %[[#PId2]] = OpFunctionParameter %[[#]]

!8 = !{i32 19}
!9 = !{i32 44, i32 4}
!10 = !{i32 21}
!11 = !{!8, !9}
!12 = !{}
!13 = !{!10, !15, !16}
!14 = !{!11, !12, !13}
!15 = !{i32 46, i32 16}
!16 = !{i32 47, i32 0}
