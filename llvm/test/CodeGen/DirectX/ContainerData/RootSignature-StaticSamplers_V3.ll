; RUN: opt %s -dxil-embed -dxil-globals -S -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: @dx.rts0 = private constant [248 x i8]  c"{{.*}}", section "RTS0", align 4

define void @main() #0 {
entry:
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3, i32 3 } ; function, root signature
!3 = !{ !5, !6, !7, !8 } ; list of root signature elements
!5 = !{ !"StaticSampler", i32 4, i32 2, i32 3, i32 5, float 0x3FF6CCCCC0000000, i32 9, i32 3, i32 2, float -1.280000e+02, float 1.280000e+02, i32 42, i32 0, i32 0, i32 1 }
!6 = !{ !"StaticSampler", i32 4, i32 2, i32 3, i32 5, float 0x3FF6CCCCC0000000, i32 9, i32 3, i32 2, float -1.280000e+02, float 1.280000e+02, i32 43, i32 0, i32 0, i32 2 }
!7 = !{ !"StaticSampler", i32 4, i32 2, i32 3, i32 5, float 0x3FF6CCCCC0000000, i32 9, i32 3, i32 2, float -1.280000e+02, float 1.280000e+02, i32 44, i32 0, i32 0, i32 0 }
!8 = !{ !"StaticSampler", i32 4, i32 2, i32 3, i32 5, float 0x3FF6CCCCC0000000, i32 9, i32 3, i32 2, float -1.280000e+02, float 1.280000e+02, i32 45, i32 0, i32 0, i32 3 }

; DXC: - Name:            RTS0
; DXC-NEXT:     Size:            248
; DXC-NEXT:     RootSignature:
; DXC-NEXT:       Version:         3
; DXC-NEXT:       NumRootParameters: 0
; DXC-NEXT:       RootParametersOffset: 24
; DXC-NEXT:       NumStaticSamplers: 4
; DXC-NEXT:       StaticSamplersOffset: 24
; DXC-NEXT:       Parameters:      []
; DXC-NEXT:       Samplers:
; DXC-LABEL:         ShaderRegister:  42
; DXC:               SAMPLER_FLAG_UINT_BORDER_COLOR: true
; DXC-LABEL:         ShaderRegister:  43
; DXC:               SAMPLER_FLAG_NON_NORMALIZED_COORDINATES: true
; DXC-LABEL:         ShaderRegister:  44
; DXC-NOT:           SAMPLER_FLAG_NON_NORMALIZED_COORDINATES:
; DXC-NOT:           SAMPLER_FLAG_UINT_BORDER_COLOR:
; DXC-LABEL:         ShaderRegister:  45
; DXC:               SAMPLER_FLAG_UINT_BORDER_COLOR: true
; DXC-NEXT:          SAMPLER_FLAG_NON_NORMALIZED_COORDINATES: true
