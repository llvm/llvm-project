; RUN: not llc < %s 2>&1 | FileCheck %s

; CHECK: error: fast-math-flags specified for call without floating-point scalar or vector return type
define i64 @test_lrintf(float %f) {
entry:
  %0 = tail call fast i64 @llvm.lrint.i64.f32(float %f)
  ret i64 %0
}

declare i64 @llvm.lrint.i64.f32(float)
