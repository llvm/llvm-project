; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s
; This test checks if a resource is implicitly overflowing. That means, it is appending a resource after an unbounded range.  

; CHECK: error: Range UAV(register=0, space=0) cannot be appended after an unbounded range

define void @CSMain() "hlsl.shader"="compute" {
entry:
  ret void
}

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!3}
!3 = !{!"DescriptorTable", i32 0, !4, !5}
!4 = !{!"UAV", i32 -1, i32 1, i32 0, i32 2, i32 0}
!5 = !{!"UAV", i32 1, i32 0, i32 0, i32 -1, i32 0}
