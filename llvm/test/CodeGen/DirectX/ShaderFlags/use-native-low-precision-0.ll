; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

; Check that when the dx.nativelowprec module flag is set to 0, the module-level
; shader flag UseNativeLowPrecision is not set, and the MinimumPrecision feature
; flag is set due to the presence of half and i16 without native low precision.

target triple = "dxil-pc-shadermodel6.7-library"

;CHECK: ; Combined Shader Flags for Module
;CHECK-NEXT: ; Shader Flags Value: 0x00000020
;CHECK-NEXT: ;
;CHECK-NEXT: ; Note: shader requires additional functionality:
;CHECK-NEXT: ;       Minimum-precision data types
;CHECK-NEXT: ; Note: extra DXIL module flags:
;CHECK-NEXT: ;       Low-precision data types present
;CHECK-NOT:  ;       Enable native low-precision data types
;CHECK-NEXT: ;
;CHECK-NEXT: ; Shader Flags for Module Functions

;CHECK-LABEL: ; Function add_i16 : 0x00000020
define i16 @add_i16(i16 %a, i16 %b) "hlsl.export" {
  %sum = add i16 %a, %b
  ret i16 %sum
}

;CHECK-LABEL: ; Function add_i32 : 0x00000000
define i32 @add_i32(i32 %a, i32 %b) "hlsl.export" {
  %sum = add i32 %a, %b
  ret i32 %sum
}

;CHECK-LABEL: ; Function add_half : 0x00000020
define half @add_half(half %a, half %b) "hlsl.export" {
  %sum = fadd half %a, %b
  ret half %sum
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"dx.nativelowprec", i32 0}

; DXC: - Name:            SFI0
; DXC-NEXT:     Size:            8
; DXC-NEXT:     Flags:
; DXC:      MinimumPrecision: true
; DXC:      NativeLowPrecision: false
; DXC: ...
