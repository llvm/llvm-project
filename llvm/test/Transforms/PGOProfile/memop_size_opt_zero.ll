; Test to ensure the pgo memop optimization pass doesn't try to scale
; up a value profile with a 0 count, which would lead to divide by 0.
; RUN: opt < %s -passes=pgo-memop-opt -pgo-memop-count-threshold=1 -S | FileCheck %s --check-prefix=MEMOP_OPT
; RUN: opt < %s -pgo-memop-opt -pgo-memop-count-threshold=1 -S | FileCheck %s --check-prefix=MEMOP_OPT

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i8* %dst, i8* %src, i64 %conv) !prof !0 {
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %conv, i1 false), !prof !1
  ret void
}

; MEMOP_OPT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %conv, i1 false), !prof !1

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"VP", i32 1, i64 0, i64 1, i64 0, i64 2, i64 0, i64 3, i64 0, i64 9, i64 0, i64 4, i64 0, i64 5, i64 0, i64 6, i64 0, i64 7, i64 0, i64 8, i64 0}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)
