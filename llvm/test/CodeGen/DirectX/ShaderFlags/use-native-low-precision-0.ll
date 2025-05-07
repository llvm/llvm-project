; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

; Check that when the dx.nativelowprec module flag is set to 0, the module-level
; shader flag UseNativeLowPrecision is not set

target triple = "dxil-pc-shadermodel6.7-library"

;CHECK: ; Combined Shader Flags for Module
;CHECK-NEXT: ; Shader Flags Value: 0x00000020
;CHECK-NEXT: ;
;CHECK-NEXT: ; Note: shader requires additional functionality:
;CHECK-NEXT: ; Note: extra DXIL module flags:
;CHECK-NEXT: ;       Low-precision data types
;CHECK-NOT:  ;       Native 16bit types enabled
;CHECK-NEXT: ;
;CHECK-NEXT: ; Shader Flags for Module Functions

;CHECK-LABEL: ; Function add_i16 : 0x00000020
define i16 @add_i16(i16 %a, i16 %b) {
  %sum = add i16 %a, %b
  ret i16 %sum
}

;CHECK-LABEL: ; Function add_i32 : 0x00000000
define i32 @add_i32(i32 %a, i32 %b) {
  %sum = add i32 %a, %b
  ret i32 %sum
}

;CHECK-LABEL: ; Function add_half : 0x00000020
define half @add_half(half %a, half %b) {
  %sum = fadd half %a, %b
  ret half %sum
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"dx.nativelowprec", i32 0}
