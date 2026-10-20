; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.5-pixel %s | FileCheck %s

; CHECK-LABEL: define i32 @test_inner_coverage
; CHECK: %[[COV:.*]] = call i32 @dx.op.innerCoverage.i32(i32 92)
; CHECK: ret i32 %[[COV]]
define i32 @test_inner_coverage() {
entry:
  %0 = call i32 @llvm.dx.inner.coverage()
  ret i32 %0
}

; CHECK: declare i32 @dx.op.innerCoverage.i32(i32) #[[#ATTR:]]
; CHECK: attributes #[[#ATTR]] = { nounwind memory(none) }

declare i32 @llvm.dx.inner.coverage()
