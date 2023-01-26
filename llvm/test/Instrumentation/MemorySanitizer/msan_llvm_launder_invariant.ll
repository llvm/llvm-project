; Make sure MSan handles llvm.launder.invariant.group correctly.

; RUN: opt < %s -passes='module(msan),default<O1>' -msan-kernel=1 -S | FileCheck -check-prefixes=CHECK %s
; RUN: opt < %s -passes='module(msan),default<O1>' -S | FileCheck -check-prefixes=CHECK %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.Foo = type { ptr }
@flag = dso_local local_unnamed_addr global i8 0, align 1

define dso_local ptr @_Z1fv() local_unnamed_addr #0 {
entry:
  %p = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %p)
  %0 = load i8, ptr @flag, align 1
  %tobool = icmp ne i8 %0, 0
  %call = call zeroext i1 @_Z2f1PPvb(ptr nonnull %p, i1 zeroext %tobool)
  %1 = load ptr, ptr %p, align 8
  %2 = call ptr @llvm.launder.invariant.group.p0(ptr %1)
  %retval.0 = select i1 %call, ptr %2, ptr null
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %p)
  ret ptr %retval.0
}

; CHECK-NOT: call void @__msan_warning_with_origin_noreturn

declare dso_local zeroext i1 @_Z2f1PPvb(ptr, i1 zeroext) local_unnamed_addr

declare ptr @llvm.launder.invariant.group.p0(ptr)

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

attributes #0 = { sanitize_memory uwtable }
