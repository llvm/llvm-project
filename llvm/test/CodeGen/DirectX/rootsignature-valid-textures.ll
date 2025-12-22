; RUN: opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 
; "This is a valid root signature with a texture/typed buffer resource"

@TB.str = private unnamed_addr constant [3 x i8] c"TB\00", align 1

define void @CSMain() "hlsl.shader"="compute" {
entry:
  %TB =  tail call target("dx.Texture", <4 x float>, 1, 0, 0, 4) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @TB.str)
  ret void
}

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!3}
!3 = !{!"DescriptorTable", i32 0, !4}
!4 = !{!"UAV", i32 1, i32 0, i32 0, i32 -1, i32 0}
