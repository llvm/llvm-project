; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s
; CHECK: error: resource CBV (space=0, registers=[2, 2]) overlaps with resource CBV (space=0, registers=[0, 2])

define void @CSMain() "hlsl.shader"="compute" {
entry:
  ret void
}

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2, !3}
!2 = !{!"RootConstants", i32 0, i32 2, i32 0, i32 4}
!3 = !{!"DescriptorTable", i32 0, !4}
!4 = !{!"CBV", i32 3, i32 0, i32 0, i32 -1, i32 4}
