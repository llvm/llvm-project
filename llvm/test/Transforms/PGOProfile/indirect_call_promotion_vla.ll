; RUN: opt < %s -passes=pgo-icall-prom -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { i8 }
%struct.B = type { i8 }
@foo = common global ptr null, align 8

define i32 @func1(ptr %x, ...) {
entry:
  ret i32 0
}

define i32 @bar(ptr %x) {
entry:
  %tmp = load ptr, ptr @foo, align 8
; CHECK:   [[CMP:%[0-9]+]] = icmp eq ptr %tmp, @func1
; CHECK:   br i1 [[CMP]], label %if.true.direct_targ, label %if.false.orig_indirect, !prof [[BRANCH_WEIGHT:![0-9]+]]
; CHECK: if.true.direct_targ:
; CHECK:   [[DIRCALL_RET:%[0-9]+]] = call i32 (ptr, ...) @func1
; CHECK:   br label %if.end.icp
  %call = call i32 (ptr, ...) %tmp(ptr %x, i32 0), !prof !1
  ret i32 %call
}

; CHECK: [[BRANCH_WEIGHT]] = !{!"branch_weights", i32 1500, i32 100}
!1 = !{!"VP", i32 0, i64 1600, i64 -2545542355363006406, i64 1500}
