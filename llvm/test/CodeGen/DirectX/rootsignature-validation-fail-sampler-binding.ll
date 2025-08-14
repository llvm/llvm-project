; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s
; CHECK: error: register Sampler (space=2, register=3) does not have a binding in the Root Signature 

@Smp.str = private unnamed_addr constant [4 x i8] c"Smp\00", align 1


define void @CSMain() "hlsl.shader"="compute" {
entry:
  %Sampler = call target("dx.Sampler", 0) @llvm.dx.resource.handlefrombinding(i32 2, i32 3, i32 1, i32 0, i1 false, ptr nonnull @Smp.str)
  ret void
}

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2}
!2 = !{!"DescriptorTable", i32 0, !3}
!3 = !{!"Sampler", i32 1, i32 42, i32 0, i32 -1, i32 0}
