; Test basic address sanitizer instrumentation.
;
; RUN: opt < %s -passes=hwasan -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

define void @atomicrmw(i64* %ptr) sanitize_hwaddress {
; CHECK-LABEL: @atomicrmw(
; CHECK: [[PTRI8:%[^ ]*]] = bitcast i64* %ptr to i8*
; CHECK: call void @llvm.hwasan.check.memaccess({{.*}}, i8* [[PTRI8]], i32 19)
; CHECK: atomicrmw add i64* %ptr, i64 1 seq_cst
; CHECK: ret void

entry:
  %0 = atomicrmw add i64* %ptr, i64 1 seq_cst
  ret void
}

define void @cmpxchg(i64* %ptr, i64 %compare_to, i64 %new_value) sanitize_hwaddress {
; CHECK-LABEL: @cmpxchg(
; CHECK: [[PTRI8:%[^ ]*]] = bitcast i64* %ptr to i8*
; CHECK: call void @llvm.hwasan.check.memaccess({{.*}}, i8* [[PTRI8]], i32 19)
; CHECK: cmpxchg i64* %ptr, i64 %compare_to, i64 %new_value seq_cst seq_cst
; CHECK: ret void

entry:
  %0 = cmpxchg i64* %ptr, i64 %compare_to, i64 %new_value seq_cst seq_cst
  ret void
}
