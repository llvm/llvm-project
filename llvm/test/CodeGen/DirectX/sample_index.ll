; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.0-pixel"

; CHECK-LABEL: define i32 @main(
; CHECK: %[[SI:.*]] = call i32 @dx.op.sampleIndex.i32(i32 90)
; CHECK: ret i32 %[[SI]]
define i32 @main() {
entry:
  %0 = call i32 @llvm.dx.sample.index()
  ret i32 %0
}

; CHECK: declare i32 @dx.op.sampleIndex.i32(i32) #[[#ATTR:]]
; CHECK: attributes #[[#ATTR]] = { nounwind memory(none) }
