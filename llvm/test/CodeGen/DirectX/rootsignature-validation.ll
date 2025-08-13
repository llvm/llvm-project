; RUN: opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 
; expected-no-diagnostics


; Root Signature(
;   CBV(b3, space=1, visibility=SHADER_VISIBILITY_ALL)
;   DescriptorTable(SRV(t0, space=0, numDescriptors=1), visibility=SHADER_VISIBILITY_ALL)
;   DescriptorTable(Sampler(s0, numDescriptors=2), visibility=SHADER_VISIBILITY_VERTEX)
;   DescriptorTable(UAV(u0, numDescriptors=unbounded), visibility=SHADER_VISIBILITY_ALL)


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
