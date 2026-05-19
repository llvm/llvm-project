; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s
; CHECK: error: SRV register 0 in space 0 does not have a binding in the Root Signature

@SB.str = private unnamed_addr constant [3 x i8] c"SB\00", align 1

define void @CSMain() "hlsl.shader"="compute" {
entry:
; StructuredBuffer<int> In : register(t0, space0);
  %SB = tail call target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_0_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @SB.str)
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
