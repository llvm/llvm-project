; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; SampleIndex is only available to pixel shaders.
; CHECK: Cannot create SampleIndex operation: Invalid stage

target triple = "dxil-pc-shadermodel6.0-vertex"
define i32 @main() {
entry:
  %0 = call i32 @llvm.dx.sample.index()
  ret i32 %0
}
