; RUN: opt %s -dxil-embed -dxil-globals -S -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: @dx.rts0 = private constant [92 x i8]  c"{{.*}}", section "RTS0", align 4

define void @main() #0 {
entry:
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3, i32 2 } ; function, root signature
!3 = !{ !5 } ; list of root signature elements
!5 = !{ !"DescriptorTable", i32 0, !6, !7 }
!6 = !{ !"SRV", i32 1, i32 1, i32 0, i32 -1, i32 4 }
!7 = !{ !"UAV", i32 5, i32 1, i32 10, i32 5, i32 2 }

; DXC:  - Name:            RTS0
; DXC-NEXT:    Size:            92
; DXC-NEXT:    RootSignature:
; DXC-NEXT:      Version:         2
; DXC-NEXT:      NumRootParameters: 1 
; DXC-NEXT:      RootParametersOffset: 24 
; DXC-NEXT:      NumStaticSamplers: 0
; DXC-NEXT:      StaticSamplersOffset: 92
; DXC-NEXT:      Parameters:
; DXC-NEXT:        - ParameterType:   0
; DXC-NEXT:          ShaderVisibility: 0
; DXC-NEXT:          Table:
; DXC-NEXT:            NumRanges:       2
; DXC-NEXT:            RangesOffset:    44
; DXC-NEXT:            Ranges:
; DXC-NEXT:              - RangeType:       0
; DXC-NEXT:                NumDescriptors:  1
; DXC-NEXT:                BaseShaderRegister: 1
; DXC-NEXT:                RegisterSpace:   0
; DXC-NEXT:                OffsetInDescriptorsFromTableStart: 4294967295
; DXC-NEXT:                DATA_STATIC_WHILE_SET_AT_EXECUTE:   true
; DXC-NEXT:              - RangeType:       1
; DXC-NEXT:                NumDescriptors:  5
; DXC-NEXT:                BaseShaderRegister: 1
; DXC-NEXT:                RegisterSpace:   10
; DXC-NEXT:                OffsetInDescriptorsFromTableStart: 5
; DXC-NEXT:                DATA_VOLATILE: true
