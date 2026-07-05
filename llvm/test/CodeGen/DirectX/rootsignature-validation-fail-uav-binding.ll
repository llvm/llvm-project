; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s
; CHECK: error: UAV register 4294967294 in space 0 does not have a binding in the Root Signature

@RWB.str = private unnamed_addr constant [4 x i8] c"RWB\00", align 1

define void @CSMain() "hlsl.shader"="compute" {
entry:
; RWBuffer<float> UAV : register(4294967294);
  %RWB = tail call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 4294967294, i32 1, i32 0, ptr nonnull @RWB.str)
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
!8 = !{!"UAV", i32 10, i32 0, i32 0, i32 -1, i32 2}
