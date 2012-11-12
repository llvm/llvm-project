; RUN: opt %s -tailcallelim -S | FileCheck %s
; XFAIL: *

declare void @use(i8* nocapture, i8* nocapture)

define i8* @foo(i8* nocapture %A, i1 %cond) {
; CHECK: tailrecurse:
; CHECK: %A.tr = phi i8* [ %A, %0 ], [ %B, %cond_true ]
; CHECK: %cond.tr = phi i1 [ %cond, %0 ], [ false, %cond_true ]
  %B = alloca i8
; CHECK: %B = alloca i8
  br i1 %cond, label %cond_true, label %cond_false
; CHECK: br i1 %cond.tr, label %cond_true, label %cond_false
cond_true:
; CHECK: cond_true:
; CHECK: br label %tailrecurse
  call i8* @foo(i8* %B, i1 false)
  ret i8* null
cond_false:
; CHECK: cond_false
  call void @use(i8* %A, i8* %B)
; CHECK: tail call void @use(i8* %A.tr, i8* %B)
  ret i8* null
; CHECK: ret i8* null
}
