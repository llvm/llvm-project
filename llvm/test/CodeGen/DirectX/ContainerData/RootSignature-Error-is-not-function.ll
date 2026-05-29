; RUN: not opt -passes='print<dxil-root-signature>' %s -S -o - 2>&1 | FileCheck %s

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: error: First element of root signature is not a Function
; CHECK-NOT:   Definition for 'main':


define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.rootsignatures = !{!5} ; list of function/root signature pairs
!5 = !{ i32 -1, !6, i32 2 } ; function, root signature
!6 = !{ !7 } ; list of root signature elements
!7 = !{ !"RootFlags", i32 2 } ; 1 = allow_input_assembler_input_layout
