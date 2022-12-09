; Test basic address sanitizer instrumentation.
;
; RUN: opt < %s -passes=hwasan -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

define void @atomicrmw(ptr %ptr) sanitize_hwaddress {
; CHECK-LABEL: @atomicrmw(
; CHECK: call void @llvm.hwasan.check.memaccess({{.*}}, ptr %ptr, i32 19)
; CHECK: atomicrmw add ptr %ptr, i64 1 seq_cst
; CHECK: ret void

entry:
  %0 = atomicrmw add ptr %ptr, i64 1 seq_cst
  ret void
}

define void @cmpxchg(ptr %ptr, i64 %compare_to, i64 %new_value) sanitize_hwaddress {
; CHECK-LABEL: @cmpxchg(
; CHECK: call void @llvm.hwasan.check.memaccess({{.*}}, ptr %ptr, i32 19)
; CHECK: cmpxchg ptr %ptr, i64 %compare_to, i64 %new_value seq_cst seq_cst
; CHECK: ret void

entry:
  %0 = cmpxchg ptr %ptr, i64 %compare_to, i64 %new_value seq_cst seq_cst
  ret void
}
