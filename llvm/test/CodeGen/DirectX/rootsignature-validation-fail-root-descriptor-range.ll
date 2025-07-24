; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s
; CHECK: error: register UAV (space=0, register=2) is overlapping with register UAV (space=0, register=0), verify your root signature definition

define void @CSMain() "hlsl.shader"="compute" {
entry:
  ret void
}

; DescriptorTable(UAV(u0, numDescriptors=unbounded), visibility = SHADER_VISIBILITY_ALL), UAV(u2, space=0, visibility=SHADER_VISIBILITY_ALL))
!dx.rootsignatures = !{!0}
!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2, !4}
!2 = !{!"DescriptorTable", i32 2, !3}
!3 = !{!"UAV", i32 -1, i32 0, i32 0, i32 -1, i32 4}
!4 = !{!"RootUAV", i32 0, i32 2, i32 0, i32 4}
