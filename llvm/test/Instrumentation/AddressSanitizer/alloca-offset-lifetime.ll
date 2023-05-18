; Test that ASAN will not instrument lifetime markers on alloca offsets.
;
; RUN: opt < %s -passes=asan --asan-use-after-scope -S | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

%t = type { ptr, ptr, %sub, i64 }
%sub = type { i32 }

define void @foo() sanitize_address {
entry:
  %0 = alloca %t, align 8
  %x = getelementptr inbounds %t, ptr %0, i64 0, i32 2
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %x)
  call void @bar(ptr nonnull %x)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %x) #3
  ret void
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @bar(ptr)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

; CHECK: store i64 %[[STACK_BASE:.+]], ptr %asan_local_stack_base, align 8
; CHECK-NOT: store i8 0
; CHECK: call void @bar(ptr nonnull %x)
