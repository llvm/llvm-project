; RUN: split-file %s %t
; RUN: opt -S -dxil-op-lower %t/pixel.ll | FileCheck %s
; RUN: not opt -S -dxil-op-lower %t/vertex.ll 2>&1 | FileCheck %s --check-prefix=STAGE

; CHECK-LABEL: define i32 @main(
; CHECK: %[[SI:.*]] = call i32 @dx.op.sampleIndex.i32(i32 90)
; CHECK: ret i32 %[[SI]]
; CHECK: declare i32 @dx.op.sampleIndex.i32(i32) #[[#ATTR:]]
; CHECK: attributes #[[#ATTR]] = { nounwind memory(none) }

; SampleIndex is only available to pixel shaders.
; STAGE: Cannot create SampleIndex operation: Invalid stage

;--- pixel.ll
target triple = "dxil-pc-shadermodel6.0-pixel"
define i32 @main() {
entry:
  %0 = call i32 @llvm.dx.sample.index()
  ret i32 %0
}

;--- vertex.ll
target triple = "dxil-pc-shadermodel6.0-vertex"
define i32 @main() {
entry:
  %0 = call i32 @llvm.dx.sample.index()
  ret i32 %0
}
