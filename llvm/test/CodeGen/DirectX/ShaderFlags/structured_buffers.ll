; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK: ; Shader Flags Value: 0x00020000
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Raw and Structured buffers
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: {{^;$}}

%"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", { <4 x float> }, 0, 0), %struct.MyStruct }
%struct.MyStruct = type { <4 x float> }

attributes #0 = { noinline nounwind memory(none) }

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC-NEXT:       Doubles: false
; DXC-NEXT:       ComputeShadersPlusRawAndStructuredBuffers:         true
; DXC-NOT:   {{[A-Za-z]+: +true}}
; DXC:       NextUnusedBit:   false
; DXC: ...