; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-eabi | FileCheck %s

; Check if sqshl/uqshl with constant shift amout can be selected. 
define i64 @test_vqshld_s64_i(i64 %a) {
; CHECK-LABEL: test_vqshld_s64_i:
; CHECK: sqshl {{d[0-9]+}}, {{d[0-9]+}}, #36
  %1 = tail call i64 @llvm.aarch64.neon.sqshl.i64(i64 %a, i64 36)
  ret i64 %1
}

define i64 @test_vqshld_u64_i(i64 %a) {
; CHECK-LABEL: test_vqshld_u64_i:
; CHECK: uqshl {{d[0-9]+}}, {{d[0-9]+}}, #36
  %1 = tail call i64 @llvm.aarch64.neon.uqshl.i64(i64 %a, i64 36)
  ret i64 %1
}

declare i64 @llvm.aarch64.neon.uqshl.i64(i64, i64)
declare i64 @llvm.aarch64.neon.sqshl.i64(i64, i64)
