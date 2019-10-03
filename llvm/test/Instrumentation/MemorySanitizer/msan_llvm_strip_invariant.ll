; Make sure MSan handles llvm.launder.invariant.group correctly.

; RUN: opt < %s -msan -msan-kernel=1 -O1 -S | FileCheck -check-prefixes=CHECK %s
; RUN: opt < %s -msan -O1 -S | FileCheck -check-prefixes=CHECK %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@flag = dso_local local_unnamed_addr global i8 0, align 1

define dso_local i8* @f(i8* %x) local_unnamed_addr #0 {
entry:
  %0 = call i8* @llvm.strip.invariant.group.p0i8(i8* %x)
  ret i8* %0
}

; CHECK-NOT: call void @__msan_warning_noreturn

declare i8* @llvm.strip.invariant.group.p0i8(i8*)

attributes #0 = { sanitize_memory uwtable }
