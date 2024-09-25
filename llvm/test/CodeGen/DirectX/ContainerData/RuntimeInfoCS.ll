; RUN: opt %s -dxil-embed -dxil-globals -S -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC
target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: @dx.psv0 = private constant [80 x i8] c"{{.*}}", section "PSV0", align 4

define void @cs_main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="8,8,1" "hlsl.shader"="compute" }

!dx.valver = !{!0}

!0 = !{i32 1, i32 7}

; DXC: - Name:            PSV0
; DXC-NEXT:   Size:            80
; DXC-NEXT:    PSVInfo:
; DXC-NEXT:      Version:         3
; DXC-NEXT:      ShaderStage:     5
; DXC-NEXT:      MinimumWaveLaneCount: 0
; DXC-NEXT:      MaximumWaveLaneCount: 4294967295
; DXC-NEXT:      UsesViewID:      0
; DXC-NEXT:      SigInputVectors: 0
; DXC-NEXT:      SigOutputVectors: [ 0, 0, 0, 0 ]
; DXC-NEXT:      NumThreadsX:     8
; DXC-NEXT:      NumThreadsY:     8
; DXC-NEXT:      NumThreadsZ:     1
; DXC-NEXT:      EntryName:       cs_main
; DXC-NEXT:      ResourceStride:  24
; DXC-NEXT:      Resources:       []
; DXC-NEXT:      SigInputElements: []
; DXC-NEXT:      SigOutputElements: []
; DXC-NEXT:      SigPatchOrPrimElements: []
; DXC-NEXT:      InputOutputMap:
; DXC-NEXT:        - [  ]
; DXC-NEXT:        - [  ]
; DXC-NEXT:        - [  ]
; DXC-NEXT:        - [  ]
