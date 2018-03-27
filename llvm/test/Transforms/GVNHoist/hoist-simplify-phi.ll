; RUN: opt < %s -gvn-hoist -S | FileCheck %s

; This test is meant to make sure that MemorySSAUpdater works correctly
; in non-trivial cases.

; CHECK: if.else218:
; CHECK-NEXT: %0 = getelementptr inbounds %s, %s* undef, i32 0, i32 0
; CHECK-NEXT: %1 = load i32, i32* %0, align 4

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

%s = type { i32, %s**, [3 x i8], i8 }

define void @test() {
entry:
  br label %cond.end118

cond.end118:                                      ; preds = %entry
  br i1 undef, label %cleanup, label %if.end155

if.end155:                                        ; preds = %cond.end118
  br label %while.cond

while.cond:                                       ; preds = %while.body, %if.end155
  br i1 undef, label %while.end, label %while.body

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
  br i1 undef, label %if.then226, label %if.else326

if.then226:                                       ; preds = %if.else218
  %size227 = getelementptr inbounds %s, %s* undef, i32 0, i32 0
  %0 = load i32, i32* %size227, align 4
  unreachable

if.else326:                                       ; preds = %if.else218
  %size330 = getelementptr inbounds %s, %s* undef, i32 0, i32 0
  %1 = load i32, i32* %size330, align 4
  unreachable

cleanup:                                          ; preds = %while.end, %cond.end118
  ret void
}
