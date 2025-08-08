; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s
; CHECK: error: register UAV (space=1, register=3) is overlapping with register UAV (space=1, register=0), verify your root signature definition.

define void @CSMain() "hlsl.shader"="compute" {
entry:
  ret void
}

!dx.rootsignatures = !{!0}
!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2, !4}
!2 = !{!"RootUAV", i32 0, i32 3, i32 1, i32 4}
!4 = !{!"DescriptorTable", i32 0, !5}
!5 = !{!"UAV", i32 4, i32 0, i32 1, i32 -1, i32 2}
