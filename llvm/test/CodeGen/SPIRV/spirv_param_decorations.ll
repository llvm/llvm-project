; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

define spir_kernel void @k(float %a, float %b, float %c) !spirv.ParameterDecorations !14 {
entry:
  ret void
}

; CHECK-SPIRV: OpDecorate %[[#PId1:]] Restrict
; CHECK-SPIRV: OpDecorate %[[#PId1]] FPRoundingMode RTP
; CHECK-SPIRV: OpDecorate %[[#PId2:]] Volatile
; CHECK-SPIRV: %[[#PId1]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV: %[[#PId2]] = OpFunctionParameter %[[#]]

!8 = !{i32 19}
!9 = !{i32 39, i32 2}
!10 = !{i32 21}
!11 = !{!8, !9}
!12 = !{}
!13 = !{!10}
!14 = !{!11, !12, !13}
