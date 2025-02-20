; RUN: not opt -passes='print<dxil-root-signature>' %s -S -o - 2>&1 | FileCheck %s

; CHECK: error: Invalid format for Root constants element
; CHECK-NOT: Root Signature Definitions
target triple = "dxil-unknown-shadermodel6.0-compute"

define void @main() #0 {
entry:
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3 } ; function, root signature
!3 = !{ !4 } ; list of root signature elements
!4 = !{ !"RootConstants", i32 0, i32 1, i32 2 }
