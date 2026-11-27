; RUN: opt -S -passes='dxil-post-optimization-validation' %s 
; Valid scenario where shader stage is not blocked from accessing root bindings
target triple = "dxil-pc-shadermodel6.6-geometry"

%__cblayout_CB = type <{ float }>

@CB.str = private unnamed_addr constant [3 x i8] c"CB\00", align 1

define void @CSMain() "hlsl.shader"="geometry" {
entry:
  %CB = tail call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 4, 0)) @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr nonnull @CB.str)
  ret void
}
attributes #0 = { noinline nounwind "exp-shader"="cs" "hlsl.numthreads"="1,2,1" "hlsl.shader"="geometry" }

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2, !3}
!2 = !{ !"RootFlags", i32 294 } ; 294 = deny_pixel/hull/vertex/amplification_shader_root_access
!3 = !{ !"RootCBV", i32 0, i32 2, i32 0, i32 0 }
