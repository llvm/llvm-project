; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.8-vertex"

; CHECK-LABEL: define i32 @main(
; CHECK: %[[START:.*]] = call i32 @dx.op.startVertexLocation.i32(i32 256)
; CHECK: ret i32 %[[START]]
define i32 @main() {
entry:
  %0 = call i32 @llvm.dx.start.vertex.location()
  ret i32 %0
}

; CHECK: declare i32 @dx.op.startVertexLocation.i32(i32) #[[#ATTR:]]
; CHECK: attributes #[[#ATTR]] = { nounwind memory(none) }
