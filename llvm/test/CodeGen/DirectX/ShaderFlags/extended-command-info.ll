; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

; SV_StartVertexLocation and SV_StartInstanceLocation require the
; ExtendedCommandInfo feature.

; CHECK: ; Shader Flags Value: [[FLAG:0x4000000000]]
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Extended command information
; CHECK-NEXT: ; Note: extra DXIL module flags:

target triple = "dxil-pc-shadermodel6.8-vertex"

; CHECK: Function start_vertex_location : [[FLAG]]
define i32 @start_vertex_location() {
entry:
  %0 = call i32 @llvm.dx.start.vertex.location()
  ret i32 %0
}

; CHECK: Function start_instance_location : [[FLAG]]
define i32 @start_instance_location() {
entry:
  %0 = call i32 @llvm.dx.start.instance.location()
  ret i32 %0
}
