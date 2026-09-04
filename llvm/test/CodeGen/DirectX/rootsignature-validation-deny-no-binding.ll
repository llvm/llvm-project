; RUN: opt -S -passes='dxil-post-optimization-validation' %s 
; This is a valid case where no resource is being used
target triple = "dxil-pc-shadermodel6.6-pixel"

define void @CSMain() #0 {
entry:
  ret void
}
attributes #0 = { noinline nounwind "exp-shader"="cs" "hlsl.numthreads"="1,2,1" "hlsl.shader"="geometry" }

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2, !3, !4}
!2 = !{!"RootConstants", i32 0, i32 2, i32 0, i32 4}
!3 = !{ !"RootFlags", i32 294 } ; 294 = deny_pixel/hull/vertex/amplification_shader_root_access
!4 = !{ !"RootSRV", i32 0, i32 1, i32 0, i32 0 }
