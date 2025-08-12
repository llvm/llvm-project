; RUN: not opt -S -passes='dxil-post-optimization-validation' %s 2>&1 | FileCheck %s

; CHECK: error: Shader has root bindings but root signature uses a DENY flag to disallow root binding access to the shader stage.
target triple = "dxil-pc-shadermodel6.6-hull"

define void @CSMain() #0 {
entry:
  ret void
}
attributes #0 = { noinline nounwind "exp-shader"="cs" "hlsl.numthreads"="1,2,1" "hlsl.shader"="hull" }

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2}
!2 = !{ !"RootFlags", i32 294 } ; 32 = deny_pixel/hull/vertex/amplification_shader_root_access

