; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 -relocation-model=pic -frame-pointer=all -mcpu=cortex-a8 | FileCheck %s

; subs r4, #1
; cmp r4, 0
; bgt
; cmp cannot be optimized away since it will clear the overflow bit.
; gt / ge, lt, le conditions all depend on V bit.
; rdar://9172742

define i32 @t() nounwind {
; CHECK-LABEL: t:
entry:
  br label %bb2

bb:                                               ; preds = %bb2
  %0 = tail call i32 @rand() nounwind
  %1 = icmp eq i32 %0, 50
  br i1 %1, label %bb3, label %bb1

bb1:                                              ; preds = %bb
; CHECK: bb1
; CHECK: subs [[REG:r[0-9]+]], #1
  %tmp = tail call i32 @puts() nounwind
  %indvar.next = add i32 %indvar, 1
  br label %bb2

bb2:                                              ; preds = %bb1, %entry
; CHECK: cmp [[REG]], #0
; CHECK: bgt
  %indvar = phi i32 [ %indvar.next, %bb1 ], [ 0, %entry ]
  %tries.0 = sub i32 2147483647, %indvar
  %tmp1 = icmp sgt i32 %tries.0, 0
  br i1 %tmp1, label %bb, label %bb3

bb3:                                              ; preds = %bb2, %bb
  ret i32 0
}

declare i32 @rand()

declare i32 @puts() nounwind
