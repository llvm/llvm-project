; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation coverage is only valid in the pixel stage
; CHECK: in function test_coverage
; CHECK-SAME: Cannot create Coverage operation: Invalid stage

target triple = "dxil-pc-shadermodel6.7-vertex"

define i32 @test_coverage() {
entry:
  %0 = call i32 @llvm.dx.coverage()
  ret i32 %0
}

declare i32 @llvm.dx.coverage()
