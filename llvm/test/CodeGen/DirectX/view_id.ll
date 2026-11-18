; RUN: split-file %s %t
; RUN: opt -S -dxil-op-lower %t/vertex.ll | FileCheck %s
; RUN: opt -S -dxil-op-lower %t/pixel.ll | FileCheck %s

; CHECK-LABEL: define i32 @main(
; CHECK: %[[VIEW:.*]] = call i32 @dx.op.viewID.i32(i32 138)
; CHECK: ret i32 %[[VIEW]]
; CHECK: declare i32 @dx.op.viewID.i32(i32) #[[#ATTR:]]
; CHECK: attributes #[[#ATTR]] = { nounwind memory(none) }

;--- vertex.ll
target triple = "dxil-pc-shadermodel6.1-vertex"
define i32 @main() {
entry:
  %0 = call i32 @llvm.dx.view.id()
  ret i32 %0
}

;--- pixel.ll
target triple = "dxil-pc-shadermodel6.1-pixel"
define i32 @main() {
entry:
  %0 = call i32 @llvm.dx.view.id()
  ret i32 %0
}
