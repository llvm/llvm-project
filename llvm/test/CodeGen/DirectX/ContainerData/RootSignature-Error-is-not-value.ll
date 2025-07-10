; RUN: not opt -passes='print<dxil-root-signature>' %s -S -o - 2>&1 | FileCheck %s

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: error: First element of root signature is not a Value
; CHECK-NOT: Root Signature Definitions

define void @main() #0 {
entry:
  ret void
}

define void @anotherMain() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.rootsignatures = !{!2, !5} ; list of function/root signature pairs
!2 = !{ ptr @main, !3, i32 2 } ; function, root signature
!3 = !{ !4 } ; list of root signature elements
!4 = !{ !"RootFlags", i32 1 } ; 1 = allow_input_assembler_input_layout
!5 = !{ !3, !6, i32 2 } ; function, root signature
!6 = !{ !7 } ; list of root signature elements
!7 = !{ !"RootFlags", i32 2 } ; 1 = allow_input_assembler_input_layout
