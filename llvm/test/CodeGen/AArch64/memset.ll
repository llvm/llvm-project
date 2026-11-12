; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; CHECK: memset_call:
; CHECK-NOT: and
; CHECK: dup
; CHECK-NEXT: stp
; CHECK-NEXT: stp
; CHECK-NEXT: ret
define void @memset_call(ptr %0, i32 %1) {
  %3 = trunc i32 %1 to i8
  call void @llvm.memset.p0.i64(ptr %0, i8 %3, i64 64, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)

