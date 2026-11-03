; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

; Reading SV_ViewID requires the ViewID feature.

; CHECK: ; Shader Flags Value: [[FLAG:0x00200000]]
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       View Instancing
; CHECK-NEXT: ; Note: extra DXIL module flags:

target triple = "dxil-pc-shadermodel6.1-library"

; CHECK: Function view_id : [[FLAG]]
define i32 @view_id() {
entry:
  %0 = call i32 @llvm.dx.view.id()
  ret i32 %0
}

; CHECK: Function no_view_id : 0x00000000
define i32 @no_view_id() {
entry:
  ret i32 0
}
