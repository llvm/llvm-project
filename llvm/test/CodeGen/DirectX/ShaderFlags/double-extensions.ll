; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK: ; Shader Flags Value: 0x00000021
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Double-precision floating point
; CHECK-NEXT: ;       Double-precision extensions for 11.1
; CHECK-NEXT: {{^;$}}
define double @div(double %a, double %b) {
  %res = fdiv double %a, %b
  ret double %res
}


; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC-NEXT:       Doubles:         true
; DXC-NOT:   {{[A-Za-z]+: +true}}
; DXC:            DX11_1_DoubleExtensions:         true
; DXC-NOT:   {{[A-Za-z]+: +true}}
; DXC:       NextUnusedBit:   false
; DXC: ...
