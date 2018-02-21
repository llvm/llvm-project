; Test basic address sanitizer instrumentation.
;
; RUN: opt < %s -hwasan -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

define void @atomicrmw(i64* %ptr) sanitize_hwaddress {
; CHECK-LABEL: @atomicrmw(
; CHECK: lshr i64 %[[A:[^ ]*]], 56
; CHECK: call void asm sideeffect "brk #2323", "{x0}"(i64 %[[A]])
; CHECK: atomicrmw add i64* %ptr, i64 1 seq_cst
; CHECK: ret void

entry:
  %0 = atomicrmw add i64* %ptr, i64 1 seq_cst
  ret void
}

define void @cmpxchg(i64* %ptr, i64 %compare_to, i64 %new_value) sanitize_hwaddress {
; CHECK-LABEL: @cmpxchg(
; CHECK: lshr i64 %[[A:[^ ]*]], 56
; CHECK: call void asm sideeffect "brk #2323", "{x0}"(i64 %[[A]])
; CHECK: cmpxchg i64* %ptr, i64 %compare_to, i64 %new_value seq_cst seq_cst
; CHECK: ret void

entry:
  %0 = cmpxchg i64* %ptr, i64 %compare_to, i64 %new_value seq_cst seq_cst
  ret void
}
