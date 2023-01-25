; Make sure MSan handles llvm.launder.invariant.group correctly.

; RUN: opt < %s -passes='module(msan),default<O1>' -msan-kernel=1 -S | FileCheck -check-prefixes=CHECK %s
; RUN: opt < %s -passes='module(msan),default<O1>' -S | FileCheck -check-prefixes=CHECK %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@flag = dso_local local_unnamed_addr global i8 0, align 1

define dso_local ptr @f(ptr %x) local_unnamed_addr #0 {
entry:
  %0 = call ptr @llvm.strip.invariant.group.p0(ptr %x)
  ret ptr %0
}

; CHECK-NOT: call void @__msan_warning_with_origin_noreturn

declare ptr @llvm.strip.invariant.group.p0(ptr)

attributes #0 = { sanitize_memory uwtable }
