; RUN: not opt -passes='print<dxil-root-signature>' %s -S -o - 2>&1 | FileCheck %s

; CHECK: error: Invalid value for ShaderVisibility: 255
; CHECK-NOT: Root Signature Definitions

target triple = "dxil-unknown-shadermodel6.0-compute"


define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3 } ; function, root signature
!3 = !{ !5 } ; list of root signature elements
!5 = !{ !"RootConstants", i32 255, i32 1, i32 2, i32 3 }
