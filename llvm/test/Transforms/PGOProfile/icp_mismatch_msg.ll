; RUN: opt < %s -passes=pgo-icall-prom -pass-remarks-missed=pgo-icall-prom -S 2>& 1 | FileCheck %s

; CHECK: remark: <unknown>:0:0: Cannot promote indirect call to func4 with count of 1234: The number of arguments mismatch
; CHECK: remark: <unknown>:0:0: Cannot promote indirect call: target with md5sum{{.*}} not found
; CHECK: remark: <unknown>:0:0: Cannot promote indirect call to func2 with count of 7890: Return type mismatch

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = common global ptr null, align 8
@foo2 = common global ptr null, align 8
@foo3 = common global ptr null, align 8

define i32 @func4(i32 %i) {
entry:
  ret i32 %i
}

define void @func2() {
entry:
  ret void
}

define i32 @bar() {
entry:
  %tmp = load ptr, ptr @foo, align 8
  %call = call i32 %tmp(), !prof !1
  %tmp2 = load ptr, ptr @foo2, align 8
  %call1 = call i32 %tmp2(), !prof !2
  %add = add nsw i32 %call1, %call
  %tmp3 = load ptr, ptr @foo3, align 8
  %call2 = call i32 %tmp3(), !prof !3
  %add2 = add nsw i32 %add, %call2
  ret i32 %add2
}

!1 = !{!"VP", i32 0, i64 1801, i64 7651369219802541373, i64 1234, i64 -4377547752858689819, i64 567}
!2 = !{!"VP", i32 0, i64 3023, i64 -6929281286627296573, i64 2345, i64 -4377547752858689819, i64 678}
!3 = !{!"VP", i32 0, i64 7890,  i64 -4377547752858689819, i64 7890}
