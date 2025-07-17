; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s

; CHECK: error: register SRV (space=0, register=0) is not defined in Root Signature 

; Root Signature(
;   CBV(b3, space=666, visibility=SHADER_VISIBILITY_ALL)
;   DescriptorTable(SRV(t0, space=0, numDescriptors=1), visibility=SHADER_VISIBILITY_VERTEX)
;   DescriptorTable(Sampler(s0, numDescriptors=2), visibility=SHADER_VISIBILITY_ALL)
;   DescriptorTable(UAV(u0, numDescriptors=unbounded), visibility=SHADER_VISIBILITY_ALL)

@SB.str = private unnamed_addr constant [3 x i8] c"SB\00", align 1

define void @CSMain() "hlsl.shader"="compute" {
entry:
; StructuredBuffer<int> In : register(t0, space0);
  %SB = tail call target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr nonnull @SB.str)
  ret void
}

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2, !3, !5, !7}
!2 = !{!"RootCBV", i32 0, i32 3, i32 666, i32 4}
!3 = !{!"DescriptorTable", i32 1, !4}
!4 = !{!"SRV", i32 1, i32 0, i32 0, i32 -1, i32 4}
!5 = !{!"DescriptorTable", i32 0, !6}
!6 = !{!"Sampler", i32 2, i32 0, i32 0, i32 -1, i32 0}
!7 = !{!"DescriptorTable", i32 0, !8}
!8 = !{!"UAV", i32 -1, i32 0, i32 0, i32 -1, i32 2}
