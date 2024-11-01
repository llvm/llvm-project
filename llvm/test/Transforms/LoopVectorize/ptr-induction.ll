; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; This testcase causes SCEV to return a pointer-typed exit value.

; CHECK: @f
; Expect that the pointer indvar has been converted into an integer indvar.
; CHECK: %index.next = add nuw i64 %index, 4
define i32 @f(ptr readonly %a, ptr readnone %b) #0 {
entry:
  %cmp.6 = icmp ult ptr %a, %b
  br i1 %cmp.6, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %a.pn = phi ptr [ %incdec.ptr8, %while.body ], [ %a, %while.body.preheader ]
  %acc.07 = phi i32 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %incdec.ptr8 = getelementptr inbounds i32, ptr %a.pn, i64 1
  %0 = load i32, ptr %incdec.ptr8, align 1
  %add = add nuw nsw i32 %0, %acc.07
  %exitcond = icmp eq ptr %incdec.ptr8, %b
  br i1 %exitcond, label %while.cond.while.end_crit_edge, label %while.body

while.cond.while.end_crit_edge:                   ; preds = %while.body
  %add.lcssa = phi i32 [ %add, %while.body ]
  br label %while.end

while.end:                                        ; preds = %while.cond.while.end_crit_edge, %entry
  %acc.0.lcssa = phi i32 [ %add.lcssa, %while.cond.while.end_crit_edge ], [ 0, %entry ]
  ret i32 %acc.0.lcssa
}
