; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.7-library"

;CHECK: ; Combined Shader Flags for Module
;CHECK-NEXT: ; Shader Flags Value: 0x00000000
;CHECK-NEXT: ;
;CHECK-NEXT: ; Shader Flags for Module Functions
;CHECK-NEXT: ; Function add : 0x00000000

define i32 @add(i32 %a, i32 %b) {
  %sum = add i32 %a, %b
  ret i32 %sum
}
