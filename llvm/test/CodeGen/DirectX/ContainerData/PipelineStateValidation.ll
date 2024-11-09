; RUN: opt %s -dxil-embed -dxil-globals -S -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC
target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: @dx.psv0 = private constant  [76 x i8] c"{{.*}}", section "PSV0", align 4

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.valver = !{!0}

!0 = !{i32 1, i32 7}

; DXC: - Name:            PSV0
; DXC:     Size:            76
; DXC:     PSVInfo:
; DXC:       Version:         3
; DXC:       ShaderStage:     5
; DXC:       MinimumWaveLaneCount: 0
; DXC:       MaximumWaveLaneCount: 4294967295
; DXC:       UsesViewID:      0
; DXC:       SigInputVectors: 0
; DXC:       SigOutputVectors: [ 0, 0, 0, 0 ]
; DXC:       NumThreadsX:     1
; DXC:       NumThreadsY:     1
; DXC:       NumThreadsZ:     1
; DXC:       EntryName:       main
; DXC:       ResourceStride:  24
; DXC:       Resources:       []
; DXC:       SigInputElements: []
; DXC:       SigOutputElements: []
; DXC:       SigPatchOrPrimElements: []
; DXC:       InputOutputMap:
; DXC:         - [  ]
; DXC:         - [  ]
; DXC:         - [  ]
; DXC:         - [  ]