; RUN: llc -mtriple=arm64-eabi -mattr=+jsconv -o - %s | FileCheck %s

define i32 @test_jcvt(double %v) {
; CHECK-LABEL: test_jcvt:
; CHECK: fjcvtzs w0, d0
  %val = call i32 @llvm.aarch64.fjcvtzs(double %v)
  ret i32 %val
}

declare i32 @llvm.aarch64.fjcvtzs(double)
