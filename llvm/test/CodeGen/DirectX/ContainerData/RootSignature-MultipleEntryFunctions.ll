; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s

target triple = "dxil-unknown-shadermodel6.0-compute"


define void @main() {
entry:
  ret void
}

define void @anotherMain() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.rootsignatures = !{!2, !5} ; list of function/root signature pairs
!2 = !{ ptr @main, !3 } ; function, root signature
!3 = !{ !4 } ; list of root signature elements
!4 = !{ !"RootFlags", i32 1 } ; 1 = allow_input_assembler_input_layout
!5 = !{ ptr @anotherMain, !6 } ; function, root signature
!6 = !{ !7 } ; list of root signature elements
!7 = !{ !"RootFlags", i32 2 } ; 1 = allow_input_assembler_input_layout


; CHECK:  - Name:            RTS0
; CHECK-NEXT:    Size:            24
; CHECK-NEXT:    RootSignature:
; CHECK-NEXT:      Version:         2
; CHECK-NEXT:      NumParameters:   0
; CHECK-NEXT:      RootParametersOffset: 0
; CHECK-NEXT:      NumStaticSamplers: 0
; CHECK-NEXT:      StaticSamplersOffset: 0
; CHECK-NEXT:      DenyVertexShaderRootAccess: true
