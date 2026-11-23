; RUN: opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s
; A descriptor range can be placed at UINT_MAX, matching DXC's behaviour  
; CHECK-NOT: error:

define void @CSMain() "hlsl.shader"="compute" {
entry:
  ret void
}

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!3}
!3 = !{!"DescriptorTable", i32 0, !4, !5}
!4 = !{!"UAV", i32 1, i32 1, i32 0, i32 4294967294, i32 0}
!5 = !{!"UAV", i32 1, i32 0, i32 0, i32 -1, i32 0}
