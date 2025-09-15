; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-unknown-shadermodel6.0-compute"

define void @main() #0 {
entry:
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3, i32 1 } ; function, root signature
!3 = !{ !5 } ; list of root signature elements
!5 = !{ !"DescriptorTable", i32 0, !6, !7 }
!6 = !{ !"Sampler", i32 1, i32 1, i32 0, i32 -1, i32 1 }
!7 = !{ !"UAV", i32 5, i32 1, i32 10, i32 5, i32 3 }


; DXC:        - Name:            RTS0
; DXC-NEXT:     Size:            84
; DXC-NEXT:     RootSignature:
; DXC-NEXT:       Version:         1
; DXC-NEXT:       NumRootParameters: 1
; DXC-NEXT:       RootParametersOffset: 24
; DXC-NEXT:       NumStaticSamplers: 0
; DXC-NEXT:       StaticSamplersOffset: 84
; DXC-NEXT:       Parameters:
; DXC-NEXT:         - ParameterType:   DescriptorTable
; DXC-NEXT:           ShaderVisibility: All
; DXC-NEXT:           Table:
; DXC-NEXT:             NumRanges:       2
; DXC-NEXT:             RangesOffset:    44
; DXC-NEXT:             Ranges:
; DXC-NEXT:               - RangeType:       Sampler
; DXC-NEXT:                 NumDescriptors:  1
; DXC-NEXT:                 BaseShaderRegister: 1
; DXC-NEXT:                 RegisterSpace:   0
; DXC-NEXT:                 OffsetInDescriptorsFromTableStart: 4294967295
; DXC-NEXT:               - RangeType:       UAV
; DXC-NEXT:                 NumDescriptors:  5
; DXC-NEXT:                 BaseShaderRegister: 1
; DXC-NEXT:                 RegisterSpace:   10
; DXC-NEXT:                 OffsetInDescriptorsFromTableStart: 5
