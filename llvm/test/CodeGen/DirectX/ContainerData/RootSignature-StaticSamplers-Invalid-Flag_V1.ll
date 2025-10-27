; RUN: not opt -passes='print<dxil-root-signature>' %s -S -o - 2>&1 | FileCheck %s


target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: error: Invalid value for Static Sampler Flag: 1 
; CHECK-NOT: Root Signature Definitions

define void @main() #0 {
entry:
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3, i32 1 } ; function, root signature
!3 = !{ !5 } ; list of root signature elements
!5 = !{ !"StaticSampler", i32 4, i32 2, i32 3, i32 5, float 0x3FF6CCCCC0000000, i32 9, i32 3, i32 2, float -1.280000e+02, float 1.280000e+02, i32 42, i32 0, i32 0, i32 1 }
