; RUN: opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 
; We have a valid root signature, this should compile successfully

define void @CSMain() "hlsl.shader"="compute" {
entry:
  ret void
}

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2, !3, !5, !7, !9}
!2 = !{!"RootCBV", i32 0, i32 3, i32 1, i32 4}
!9 = !{!"RootConstants", i32 0, i32 2, i32 0, i32 4}
!3 = !{!"DescriptorTable", i32 0, !4}
!4 = !{!"SRV", i32 1, i32 0, i32 0, i32 -1, i32 0}
!5 = !{!"DescriptorTable", i32 0, !6}
!6 = !{!"Sampler", i32 5, i32 3, i32 2, i32 -1, i32 0}
!7 = !{!"DescriptorTable", i32 0, !8}
!8 = !{!"UAV", i32 -1, i32 0, i32 0, i32 -1, i32 2}
