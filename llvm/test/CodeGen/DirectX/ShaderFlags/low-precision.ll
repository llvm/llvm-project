; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-pc-shadermodel6.7-library"

;CHECK: ; Combined Shader Flags for Module
;CHECK-NEXT: ; Shader Flags Value: 0x00000020
;CHECK-NEXT: ;
;CHECK-NEXT: ; Note: shader requires additional functionality:
;CHECK-NEXT: ; Note: extra DXIL module flags:
;CHECK-NEXT: ;       D3D11_1_SB_GLOBAL_FLAG_ENABLE_MINIMUM_PRECISION
;CHECK-NEXT: ;
;CHECK-NEXT: ; Shader Flags for Module Functions

;CHECK-LABEL: ; Function add_i16 : 0x00000020
define i16 @add_i16(i16 %a, i16 %b) #0 {
  %sum = add i16 %a, %b
  ret i16 %sum
}

;CHECK-LABEL: ; Function add_i32 : 0x00000000
define i32 @add_i32(i32 %a, i32 %b) #0 {
  %sum = add i32 %a, %b
  ret i32 %sum
}

;CHECK-LABEL: ; Function add_half : 0x00000020
define half @add_half(half %a, half %b) #0 {
  %sum = fadd half %a, %b
  ret half %sum
}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NOT:     Flags:
; DXC: ...
