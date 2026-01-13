; RUN: opt %s -dxil-embed -dxil-globals -S -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: @dx.rts0 = private constant [48 x i8]  c"{{.*}}", section "RTS0", align 4

define void @main() #0 {
entry:
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3, i32 2 } ; function, root signature
!3 = !{ !5 } ; list of root signature elements
!5 = !{ !"RootConstants", i32 0, i32 1, i32 2, i32 3 }

; DXC:  - Name:            RTS0
; DXC-NEXT:    Size:            48
; DXC-NEXT:    RootSignature:
; DXC-NEXT:      Version:         2
; DXC-NEXT:      NumRootParameters: 1
; DXC-NEXT:      RootParametersOffset: 24
; DXC-NEXT:      NumStaticSamplers: 0
; DXC-NEXT:      StaticSamplersOffset: 48
; DXC-NEXT:      Parameters:
; DXC-NEXT:        - ParameterType:   Constants32Bit
; DXC-NEXT:          ShaderVisibility: All
; DXC-NEXT:          Constants:
; DXC-NEXT:            Num32BitValues:  3
; DXC-NEXT:            RegisterSpace:   2
; DXC-NEXT:            ShaderRegister:  1
