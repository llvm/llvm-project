; RUN: opt < %s -passes=gvn-hoist -S | FileCheck %s

; This test is meant to make sure that MemorySSAUpdater works correctly
; in non-trivial cases.

; CHECK: if.else218:
; CHECK-NEXT: %0 = load i32, ptr undef, align 4

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

%s = type { i32, ptr, [3 x i8], i8 }

define void @test(i1 %arg) {
entry:
  br label %cond.end118

cond.end118:                                      ; preds = %entry
  br i1 %arg, label %cleanup, label %if.end155

if.end155:                                        ; preds = %cond.end118
  br label %while.cond

while.cond:                                       ; preds = %while.body, %if.end155
  br i1 %arg, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  br label %while.cond

while.end:                                        ; preds = %while.cond
  switch i32 undef, label %if.else218 [
    i32 1, label %cleanup
    i32 0, label %if.then174
  ]

if.then174:                                       ; preds = %while.end
  unreachable

if.else218:                                       ; preds = %while.end
  br i1 %arg, label %if.then226, label %if.else326

if.then226:                                       ; preds = %if.else218
  %0 = load i32, ptr undef, align 4
  unreachable

if.else326:                                       ; preds = %if.else218
  %1 = load i32, ptr undef, align 4
  unreachable

cleanup:                                          ; preds = %while.end, %cond.end118
  ret void
}
