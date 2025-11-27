; RUN: opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s
; This is a valid code, it checks the limits of a binding space
; CHECK-NOT: error:

%__cblayout_CB = type <{ float }>

@CB.str = private unnamed_addr constant [3 x i8] c"CB\00", align 1

define void @CSMain() "hlsl.shader"="compute" {
entry:
  %CB = tail call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 4, 0)) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 5, i32 0, ptr nonnull @CB.str)
  ret void
}

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2}
!2 = !{!"DescriptorTable", i32 0, !3}
!3 = !{!"CBV", i32 5, i32 0, i32 0, i32 0, i32 4}
