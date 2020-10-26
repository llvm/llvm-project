; Test that ASAN will not instrument lifetime markers on alloca offsets.
;
; RUN: opt < %s --asan --asan-use-after-scope -S -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes=asan-function-pipeline --asan-use-after-scope -S | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

%t = type { void (%t*)*, void (%t*)*, %sub, i64 }
%sub = type { i32 }

define void @foo() sanitize_address {
entry:
  %0 = alloca %t, align 8
  %x = getelementptr inbounds %t, %t* %0, i64 0, i32 2
  %1 = bitcast %sub* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %1)
  call void @bar(%sub* nonnull %x)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %1) #3
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare void @bar(%sub*)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

; CHECK: store i64 %[[STACK_BASE:.+]], i64* %asan_local_stack_base, align 8
; CHECK-NOT: store i8 0
; CHECK: call void @bar(%sub* nonnull %x)
