; RUN: opt %s -dxil-embed -dxil-globals -S -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-unknown-shadermodel1.1-rootsignature"

; CHECK: @dx.rts0 = private constant [24 x i8]  c"{{.*}}", section "RTS0", align 4

!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ null, !3, i32 2 } ; function, root signature, version
!3 = !{ !4 } ; list of root signature elements
!4 = !{ !"RootFlags", i32 1 } ; 1 = allow_input_assembler_input_layout

; DXC:       - Name:            RTS0
; DXC-NEXT:    Size:            24
; DXC-NEXT:    RootSignature:
; DXC-NEXT:      Version:         2
; DXC-NEXT:      NumRootParameters: 0
; DXC-NEXT:      RootParametersOffset: 24
; DXC-NEXT:      NumStaticSamplers: 0
; DXC-NEXT:      StaticSamplersOffset: 24
; DXC-NEXT:      Parameters: []
; DXC-NEXT:      AllowInputAssemblerInputLayout: true

