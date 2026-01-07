; RUN: not opt -passes='print<dxil-root-signature>' %s -S -o - 2>&1 | FileCheck %s

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: error: Invalid value for DescriptorFlag: 66666
; CHECK-NOT: Root Signature Definitions

define void @main() #0 {
entry:
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3, i32 2 } ; function, root signature
!3 = !{ !5 } ; list of root signature elements
!5 = !{ !"DescriptorTable", i32 0, !6, !7 }
!6 = !{ !"SRV", i32 1, i32 1, i32 0, i32 -1, i32 66666 }
!7 = !{ !"UAV", i32 5, i32 1, i32 10, i32 5, i32 2 }
