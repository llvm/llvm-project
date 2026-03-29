; RUN: opt -S -passes=pre-isel-intrinsic-lowering -mtriple=x86_64-unknown-linux -mcpu=haswell < %s | FileCheck %s --check-prefix=HASWELL
; RUN: opt -S -passes=pre-isel-intrinsic-lowering -mtriple=x86_64-unknown-linux -mcpu=x86-64 < %s | FileCheck %s --check-prefix=GENERIC

; Verify that PreISelIntrinsicLowering acts as a safety net and replaces
; any surviving target intrinsics with constants.

define i1 @test_safety_has_feature() {
; HASWELL-LABEL: @test_safety_has_feature(
; HASWELL:       ret i1 true
;
; GENERIC-LABEL: @test_safety_has_feature(
; GENERIC:       ret i1 false
  %1 = call i1 @llvm.target.has.feature(metadata !"avx2")
  ret i1 %1
}

define i1 @test_safety_is_cpu() {
; HASWELL-LABEL: @test_safety_is_cpu(
; HASWELL:       ret i1 true
;
; GENERIC-LABEL: @test_safety_is_cpu(
; GENERIC:       ret i1 false
  %1 = call i1 @llvm.target.is.cpu(metadata !"haswell")
  ret i1 %1
}

declare i1 @llvm.target.has.feature(metadata)
declare i1 @llvm.target.is.cpu(metadata)
