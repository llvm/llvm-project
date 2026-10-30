; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-pixel %s | FileCheck %s

; CHECK-LABEL: define i32 @test_coverage
; CHECK: %[[COV:.*]] = call i32 @dx.op.coverage.i32(i32 91)
; CHECK: ret i32 %[[COV]]
define i32 @test_coverage() {
entry:
  %0 = call i32 @llvm.dx.coverage()
  ret i32 %0
}

; CHECK: declare i32 @dx.op.coverage.i32(i32) #[[#ATTR:]]
; CHECK: attributes #[[#ATTR]] = { nounwind memory(none) }

declare i32 @llvm.dx.coverage()
