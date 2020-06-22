; Make sure MSan handles llvm.launder.invariant.group correctly.

; RUN: opt < %s -msan -msan-kernel=1 -O1 -S | FileCheck -check-prefixes=CHECK %s
; RUN: opt < %s -msan -O1 -S | FileCheck -check-prefixes=CHECK %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.Foo = type { i32 (...)** }
@flag = dso_local local_unnamed_addr global i8 0, align 1

define dso_local %class.Foo* @_Z1fv() local_unnamed_addr #0 {
entry:
  %p = alloca i8*, align 8
  %0 = bitcast i8** %p to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0)
  %1 = load i8, i8* @flag, align 1
  %tobool = icmp ne i8 %1, 0
  %call = call zeroext i1 @_Z2f1PPvb(i8** nonnull %p, i1 zeroext %tobool)
  %2 = load i8*, i8** %p, align 8
  %3 = call i8* @llvm.launder.invariant.group.p0i8(i8* %2)
  %4 = bitcast i8* %3 to %class.Foo*
  %retval.0 = select i1 %call, %class.Foo* %4, %class.Foo* null
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0)
  ret %class.Foo* %retval.0
}

; CHECK-NOT: call void @__msan_warning_with_origin_noreturn

declare dso_local zeroext i1 @_Z2f1PPvb(i8**, i1 zeroext) local_unnamed_addr

declare i8* @llvm.launder.invariant.group.p0i8(i8*)

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

attributes #0 = { sanitize_memory uwtable }
