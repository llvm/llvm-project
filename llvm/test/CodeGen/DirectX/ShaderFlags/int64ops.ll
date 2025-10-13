; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-pc-shadermodel6.7-library"

;CHECK: ; Combined Shader Flags for Module
;CHECK-NEXT: ; Shader Flags Value: 0x00100000
;CHECK-NEXT: ;
;CHECK-NEXT: ; Note: shader requires additional functionality:
;CHECK-NEXT: ;       64-Bit integer
;CHECK-NEXT: ; Note: extra DXIL module flags:
;CHECK-NEXT: ;
;CHECK-NEXT: ; Shader Flags for Module Functions
;CHECK-NEXT: ; Function add : 0x00100000

define i64 @add(i64 %a, i64 %b) #0 {
  %sum = add i64 %a, %b
  ret i64 %sum
}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC-NOT:   {{[A-Za-z]+: +true}}
; DXC:            Int64Ops:        true
; DXC-NOT:   {{[A-Za-z]+: +true}}
; DXC:       NextUnusedBit:   false
; DXC: ...
