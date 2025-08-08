; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s
; CHECK: error: register UAV (space=10, register=4294967295) is overlapping with register UAV (space=10, register=4294967295), verify your root signature definition.
define void @CSMain() "hlsl.shader"="compute" {
entry:
  ret void
}

; DescriptorTable(UAV(u0, numDescriptors=unbounded), visibility = SHADER_VISIBILITY_HULL), DescriptorTable(UAV(u2, numDescriptors=4))
!dx.rootsignatures = !{!0}
!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2, !4}
!2 = !{!"DescriptorTable", i32 0, !3}
!3 = !{!"UAV", i32 -1, i32 -1, i32 10, i32 -1, i32 2}
!4 = !{!"DescriptorTable", i32 0, !5}
!5 = !{ !"UAV", i32 -1, i32 -1, i32 10, i32 5, i32 2 }
