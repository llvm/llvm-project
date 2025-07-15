; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s
; CHECK: error: register CB (space=666, register=2) is not defined in Root Signature
; Root Signature(RootConstants(num32BitConstants=4, b2))

%__cblayout_CB = type <{ float }>

@CB.str = private unnamed_addr constant [3 x i8] c"CB\00", align 1

define void @CSMain() "hlsl.shader"="compute" {
entry:
; cbuffer CB : register(b2, space666) {
;  float a;
; }
  %CB = tail call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 4, 0)) @llvm.dx.resource.handlefrombinding(i32 666, i32 2, i32 1, i32 0, i1 false, ptr nonnull @CB.str)
  ret void
}

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2}
!2 = !{!"RootConstants", i32 0, i32 2, i32 0, i32 4}
