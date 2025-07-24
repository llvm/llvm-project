; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 | FileCheck %s
; CHECK: error: register Sampler (space=0, register=42) is overlapping with register Sampler (space=0, register=42), verify your root signature definition.

define void @CSMain() "hlsl.shader"="compute" {
entry:
  ret void
}

; RootConstants(num32BitConstants=4, b2), DescriptorTable(CBV(b0, numDescriptors=3))
!dx.rootsignatures = !{!0}
!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2, !3}
!2 = !{ !"StaticSampler", i32 4, i32 2, i32 3, i32 5, float 0x3FF6CCCCC0000000, i32 9, i32 3, i32 2, float -1.280000e+02, float 1.280000e+02, i32 42, i32 0, i32 0 }
!3 = !{ !"StaticSampler", i32 4, i32 2, i32 3, i32 5, float 0x3FF6CCCCC0000000, i32 9, i32 3, i32 2, float -1.280000e+02, float 1.280000e+02, i32 42, i32 0, i32 0 }
