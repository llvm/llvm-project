; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK: ; Shader Flags Value: 0x00000001
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Double-precision floating point
; CHECK-NEXT: {{^;$}}
define double @add(double %a, double %b) {
  %sum = fadd double %a, %b
  ret double %sum
}

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            16
; DXC-NEXT:     Flags:
; DXC-NEXT:       Doubles:         true
; DXC-NOT:   {{[A-Za-z]+: +true}}
; DXC:       NextUnusedBit:   false
; DXC: ...
