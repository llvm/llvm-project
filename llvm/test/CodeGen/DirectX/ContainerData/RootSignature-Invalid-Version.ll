; RUN: not opt -passes='print<dxil-root-signature>' %s -S -o - 2>&1 | FileCheck %s

target triple = "dxil-unknown-shadermodel6.0-compute"


; CHECK: error: Invalid value for Version: 4
; CHECK-NOT: Root Signature Definitions
define void @main() #0 {
entry:
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2, !3, !4, !5} ; list of function/root signature pairs
!2 = !{ ptr @main, !6, i32 1 } ; function, root signature
!3 = !{ ptr @main, !6, i32 4 } ; function, root signature
!4 = !{ ptr @main, !6, i32 2 } ; function, root signature
!5 = !{ ptr @main, !6, i32 3 } ; function, root signature
!6 = !{ } ; list of root signature elements
