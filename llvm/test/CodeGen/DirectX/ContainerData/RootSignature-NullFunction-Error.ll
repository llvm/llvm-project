; RUN: not opt -passes='print<dxil-root-signature>' %s -S -o - 2>&1 | FileCheck %s

; CHECK: error: Function associated with Root Signature definition is null
; CHECK-NOT: Root Signature Definitions

target triple = "dxil-unknown-shadermodel6.0-compute"

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.rootsignatures = !{!2, !5} ; list of function/root signature pairs
!2 = !{ ptr @main, !3 } ; function, root signature
!3 = !{ !4 } ; list of root signature elements
!4 = !{ !"RootFlags", i32 1 } ; 1 = allow_input_assembler_input_layout
!5 = !{ null, !6 } ; function, root signature
!6 = !{ !7 } ; list of root signature elements
!7 = !{ !"RootFlags", i32 2 } ; 1 = allow_input_assembler_input_layout
