; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s

target triple = "dxil-pc-shadermodel6.7-library"
define double @div(double %a, double %b) #0 {
  %res = fdiv double %a, %b
  ret double %res
}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}

; CHECK: - Name:            SFI0
; CHECK-NEXT:     Size:            8
; CHECK-NEXT:     Flags:
; CHECK:       Doubles:         true
; CHECK:       DX11_1_DoubleExtensions:         true

