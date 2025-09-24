; RUN: opt -passes='print<dxil-root-signature>' %s -S -o - 2>&1 | FileCheck %s

target triple = "dxil-unknown-shadermodel6.0-compute"


define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3, i32 2 } ; function, root signature
!3 = !{ !4, !5, !6, !7 } ; list of root signature elements
!4 = !{ !"RootFlags", i32 1 } ; 1 = allow_input_assembler_input_layout
!5 = !{ !"RootConstants", i32 0, i32 1, i32 2, i32 3 }
!6 = !{ !"RootSRV", i32 1, i32 4, i32 5, i32 4 }
!7 = !{ !"DescriptorTable", i32 0, !8, !9 }
!8 = !{ !"SRV", i32 1, i32 1, i32 0, i32 -1, i32 4 }
!9 = !{ !"UAV", i32 5, i32 1, i32 10, i32 5, i32 2 }

;CHECK-LABEL: Definition for 'main':
;CHECK-NEXT:  Flags: 0x000001
;CHECK-NEXT:  Version: 2
;CHECK-NEXT:  RootParametersOffset: 24
;CHECK-NEXT:  NumParameters: 3
;CHECK-NEXT:   - Parameter Type: Constants32Bit
;CHECK-NEXT:     Shader Visibility: All
;CHECK-NEXT:     Register Space: 2
;CHECK-NEXT:     Shader Register: 1
;CHECK-NEXT:     Num 32 Bit Values: 3
;CHECK-NEXT:   - Parameter Type: SRV
;CHECK-NEXT:     Shader Visibility: Vertex
;CHECK-NEXT:     Register Space: 5
;CHECK-NEXT:     Shader Register: 4
;CHECK-NEXT:     Flags: 4
;CHECK-NEXT:   - Parameter Type: DescriptorTable
;CHECK-NEXT:     Shader Visibility: All
;CHECK-NEXT:     NumRanges: 2
;CHECK-NEXT:     - Range Type: SRV
;CHECK-NEXT:       Register Space: 0
;CHECK-NEXT:       Base Shader Register: 1
;CHECK-NEXT:       Num Descriptors: 1
;CHECK-NEXT:       Offset In Descriptors From Table Start: 4294967295
;CHECK-NEXT:       Flags: 4
;CHECK-NEXT:     - Range Type: UAV
;CHECK-NEXT:       Register Space: 10
;CHECK-NEXT:       Base Shader Register: 1
;CHECK-NEXT:       Num Descriptors: 5
;CHECK-NEXT:       Offset In Descriptors From Table Start: 5
;CHECK-NEXT:       Flags: 2
;CHECK-NEXT:  NumStaticSamplers: 0
;CHECK-NEXT:  StaticSamplersOffset: 0
