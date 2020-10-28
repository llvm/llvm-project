; RUN: opt < %s -jump-threading -S -verify | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = global i32 0, align 4

define void @foo(i32 %cond1, i32 %cond2) {
; CHECK-LABEL: @foo
; CHECK-LABEL: entry
entry:
  %tobool = icmp eq i32 %cond1, 0
  br i1 %tobool, label %bb.cond2, label %bb.f1

bb.f1:
  call void @f1()
  br label %bb.cond2
; Verify that we branch on cond2 without checking ptr.
; CHECK:      call void @f1()
; CHECK-NEXT: icmp eq i32 %cond2, 0
; CHECK-NEXT: label %bb.f4, label %bb.f2

bb.cond2:
  %ptr = phi i32* [ null, %bb.f1 ], [ @a, %entry ]
  %tobool1 = icmp eq i32 %cond2, 0
  br i1 %tobool1, label %bb.file, label %bb.f2
; Verify that we branch on cond2 without checking ptr.
; CHECK:      icmp eq i32 %cond2, 0
; CHECK-NEXT: label %bb.f3, label %bb.f2

bb.f2:
  call void @f2()
  br label %exit

; Verify that we eliminate this basic block.
; CHECK-NOT: bb.file:
bb.file:
  %cmp = icmp eq i32* %ptr, null
  br i1 %cmp, label %bb.f4, label %bb.f3

bb.f3:
  call void @f3()
  br label %exit

bb.f4:
  call void @f4()
  br label %exit

exit:
  ret void
}

declare void @f1()

declare void @f2()

declare void @f3()

declare void @f4()


define void @foo2(i32 %cond1, i32 %cond2) {
; CHECK-LABEL: @foo2
; CHECK-LABEL: entry
entry:
  %tobool = icmp ne i32 %cond1, 0
  br i1 %tobool, label %bb.f1, label %bb.f2

bb.f1:
  call void @f1()
  br label %bb.cond2
; Verify that we branch on cond2 without checking tobool again.
; CHECK:      call void @f1()
; CHECK-NEXT: icmp eq i32 %cond2, 0
; CHECK-NEXT: label %exit, label %bb.f3

bb.f2:
  call void @f2()
  br label %bb.cond2
; Verify that we branch on cond2 without checking tobool again.
; CHECK:      call void @f2()
; CHECK-NEXT: icmp eq i32 %cond2, 0
; CHECK-NEXT: label %exit, label %bb.f4

bb.cond2:
  %tobool1 = icmp eq i32 %cond2, 0
  br i1 %tobool1, label %exit, label %bb.cond1again

; Verify that we eliminate this basic block.
; CHECK-NOT: bb.cond1again:
bb.cond1again:
  br i1 %tobool, label %bb.f3, label %bb.f4

bb.f3:
  call void @f3()
  br label %exit

bb.f4:
  call void @f4()
  br label %exit

exit:
  ret void
}


; Verify that we do *not* thread any edge.  We used to evaluate
; constant expressions like:
;
;   icmp ugt i8* null, inttoptr (i64 4 to i8*)
;
; as "true", causing jump threading to a wrong destination.
define void @foo3(i8* %arg1, i8* %arg2) {
; CHECK-LABEL: @foo
; CHECK-NOT: bb_{{[^ ]*}}.thread:
entry:
  %cmp1 = icmp eq i8* %arg1, null
  br i1 %cmp1, label %bb_bar1, label %bb_end

bb_bar1:
  call void @bar(i32 1)
  br label %bb_end

bb_end:
  %cmp2 = icmp ne i8* %arg2, null
  br i1 %cmp2, label %bb_cont, label %bb_bar2

bb_bar2:
  call void @bar(i32 2)
  br label %bb_exit

bb_cont:
  %cmp3 = icmp ule i8* %arg1, inttoptr (i64 4 to i8*)
  br i1 %cmp3, label %bb_exit, label %bb_bar3

bb_bar3:
  call void @bar(i32 3)
  br label %bb_exit

bb_exit:
  ret void
}

declare void @bar(i32)


;; Test that we skip unconditional PredBB when threading jumps through two
;; successive basic blocks.

define i32 @foo4(i32* %0) {
; CHECK-LABEL: @f
; CHECK: br i1 %good, label %pred.bb, label %pred.pred.bb
entry:
  %size = call i64 @get_size(i32* %0)
  %good = icmp ugt i64 %size, 3
  br i1 %good, label %pred.bb, label %pred.pred.bb

; CHECK:      pred.pred.bb:
; CHECK:       br label %pred.bb
; CHECK:      pred.bb:
; CHECK:       br label %bb
; CHECK:      bb:
pred.pred.bb:                                        ; preds = %entry
  call void @effect()
  br label %pred.bb
pred.bb:                                             ; preds = %pred.pred.bb, %entry
  %v = load i32, i32* %0
  br label %bb

bb:                                                  ; preds = %pred.bb
  call void @effect1(i8* blockaddress(@foo4, %bb))
  br i1 %good, label %cont2, label %cont1

cont1:                                               ; preds = %bb
  br i1 %good, label %exit, label %cont2
cont2:                                               ; preds = %bb
  br label %exit
exit:                                                ; preds = %cont1, %cont2
  ret i32 %v
}

declare i64 @get_size(i32*)
declare void @effect()
declare void @effect1(i8*)
