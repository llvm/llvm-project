; RUN: opt %s -dxil-embed -dxil-globals -S -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: @dx.rts0 = private constant [76 x i8]  c"{{.*}}", section "RTS0", align 4

define void @main() #0 {
entry:
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3, i32 2 } ; function, root signature
!3 = !{ !5 } ; list of root signature elements
!5 = !{ !"StaticSampler", i32 4, i32 2, i32 3, i32 5, float 0x3FF6CCCCC0000000, i32 9, i32 3, i32 2, float -1.280000e+02, float 1.280000e+02, i32 42, i32 0, i32 0 }

; DXC: - Name:            RTS0
; DXC-NEXT:     Size:            76
; DXC-NEXT:     RootSignature:
; DXC-NEXT:       Version:         2
; DXC-NEXT:       NumRootParameters: 0
; DXC-NEXT:       RootParametersOffset: 24
; DXC-NEXT:       NumStaticSamplers: 1
; DXC-NEXT:       StaticSamplersOffset: 24
; DXC-NEXT:       Parameters:      []
; DXC-NEXT:       Samplers:
; DXC-NEXT:         - Filter:          MinPointMagLinearMipPoint
; DXC-NEXT:           AddressU:        Mirror
; DXC-NEXT:           AddressV:        Clamp
; DXC-NEXT:           AddressW:        MirrorOnce
; DXC-NEXT:           MipLODBias:      1.425
; DXC-NEXT:           MaxAnisotropy:   9
; DXC-NEXT:           ComparisonFunc:  Equal
; DXC-NEXT:           BorderColor:     OpaqueWhite
; DXC-NEXT:           MinLOD:          -128
; DXC-NEXT:           MaxLOD:          128
; DXC-NEXT:           ShaderRegister:  42
; DXC-NEXT:           RegisterSpace:   0
; DXC-NEXT:           ShaderVisibility: All
