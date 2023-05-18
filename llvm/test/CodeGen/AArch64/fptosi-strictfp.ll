; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

define i128 @test_fixtfti(fp128 %ld) #0 {
; CHECK-LABEL: test_fixtfti:
; CHECK: bl	__fixtfti
entry:
  %conv = call i128 @llvm.experimental.constrained.fptosi.i128.f128(fp128 %ld, metadata !"fpexcept.strict") #0
  ret i128 %conv
}

declare i128 @llvm.experimental.constrained.fptosi.i128.f128(fp128, metadata)

define i128 @test_fixtftu(fp128 %ld) #0 {
; CHECK-LABEL: test_fixtftu:
; CHECK: bl	__fixunstfti
entry:
  %conv = call i128 @llvm.experimental.constrained.fptoui.i128.f128(fp128 %ld, metadata !"fpexcept.strict") #0
  ret i128 %conv
}

declare i128 @llvm.experimental.constrained.fptoui.i128.f128(fp128, metadata)

attributes #0 = { strictfp }
