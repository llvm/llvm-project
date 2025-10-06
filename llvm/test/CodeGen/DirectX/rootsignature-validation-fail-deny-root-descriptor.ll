; RUN: not opt -S -passes='dxil-post-optimization-validation' %s 2>&1 | FileCheck %s

; CHECK: error: Shader has root bindings but root signature uses a DENY flag to disallow root binding access to the shader stage.
target triple = "dxil-pc-shadermodel6.6-pixel"

@SB.str = private unnamed_addr constant [3 x i8] c"SB\00", align 1

define void @CSMain() "hlsl.shader"="pixel" {
entry:
  %SB = tail call target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @SB.str)
  ret void
}

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2, !3}
!2 = !{!"DescriptorTable", i32 0, !4}
!4 = !{!"SRV", i32 1, i32 0, i32 0, i32 -1, i32 4}
!3 = !{!"RootFlags", i32 32} ; 32 = deny_pixel_shader_root_access
