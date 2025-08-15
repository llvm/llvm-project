; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s
; CHECK: error: resource TB at register (space=0, register=0) is bound to a texture or typed buffer.

@TB.str = private unnamed_addr constant [3 x i8] c"TB\00", align 1

define void @CSMain() "hlsl.shader"="compute" {
entry:
  %TB =  tail call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr nonnull @TB.str)
  ret void
}

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2}
!2 = !{!"RootUAV", i32 0, i32 0, i32 0, i32 4}
