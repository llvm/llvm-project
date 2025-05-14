; RUN: opt -passes='print<dxil-root-signature>' %s -S -o - 2>&1 | FileCheck %s

target triple = "dxil-unknown-shadermodel6.0-compute"


define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3 } ; function, root signature
!3 = !{ !4, !5 } ; list of root signature elements
!4 = !{ !"RootFlags", i32 1 } ; 1 = allow_input_assembler_input_layout
!5 = !{ !"RootConstants", i32 0, i32 1, i32 2, i32 3 }

;CHECK-LABEL: Definition for 'main':
;CHECK-NEXT:  Flags: 0x000001
;CHECK-NEXT:  Version: 2
;CHECK-NEXT:  RootParametersOffset: 24
;CHECK-NEXT:  NumParameters: 1
;CHECK-NEXT:   - Parameter Type: 1
;CHECK-NEXT:     Shader Visibility: 0
;CHECK-NEXT:     Register Space: 2
;CHECK-NEXT:     Shader Register: 1
;CHECK-NEXT:     Num 32 Bit Values: 3
;CHECK-NEXT:  NumStaticSamplers: 0
;CHECK-NEXT:  StaticSamplersOffset: 0
