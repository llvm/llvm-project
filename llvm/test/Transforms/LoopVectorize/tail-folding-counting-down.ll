; RUN: opt < %s -passes=loop-vectorize -prefer-predicate-over-epilogue=predicate-dont-vectorize -force-vector-width=4 -S | FileCheck %s

; Check that a counting-down loop which has no primary induction variable
; is vectorized with preferred predication.

; CHECK-LABEL: vector.body:
; CHECK-LABEL: middle.block:
; CHECK-NEXT:    br i1 true,

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

define dso_local void @foo(ptr noalias nocapture readonly %A, ptr noalias nocapture readonly %B, ptr noalias nocapture %C, i32 %N) {
entry:
  %cmp6 = icmp eq i32 %N, 0
  br i1 %cmp6, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %N.addr.010 = phi i32 [ %dec, %while.body ], [ %N, %while.body.preheader ]
  %C.addr.09 = phi ptr [ %incdec.ptr4, %while.body ], [ %C, %while.body.preheader ]
  %B.addr.08 = phi ptr [ %incdec.ptr1, %while.body ], [ %B, %while.body.preheader ]
  %A.addr.07 = phi ptr [ %incdec.ptr, %while.body ], [ %A, %while.body.preheader ]
  %incdec.ptr = getelementptr inbounds i8, ptr %A.addr.07, i32 1
  %0 = load i8, ptr %A.addr.07, align 1
  %incdec.ptr1 = getelementptr inbounds i8, ptr %B.addr.08, i32 1
  %1 = load i8, ptr %B.addr.08, align 1
  %add = add i8 %1, %0
  %incdec.ptr4 = getelementptr inbounds i8, ptr %C.addr.09, i32 1
  store i8 %add, ptr %C.addr.09, align 1
  %dec = add i32 %N.addr.010, -1
  %cmp = icmp eq i32 %dec, 0
  br i1 %cmp, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  ret void
}

; Make sure a loop is successfully vectorized with fold-tail when the backedge
; taken count is constant and used inside the loop. Issue revealed by D76992.
;
define void @reuse_const_btc(ptr %A) optsize {
; CHECK-LABEL: @reuse_const_btc
; CHECK: {{%.*}} = icmp ule <4 x i32> {{%.*}}, splat (i32 13)
; CHECK: {{%.*}} = select <4 x i1> {{%.*}}, <4 x i32> splat (i32 12), <4 x i32> splat (i32 13)
;
entry:
  br label %loop

loop:
  %riv = phi i32 [ 13, %entry ], [ %rivMinus1, %merge ]
  %sub = sub nuw nsw i32 20, %riv
  %arrayidx = getelementptr inbounds i8, ptr %A, i32 %sub
  %cond0 = icmp eq i32 %riv, 7
  br i1 %cond0, label %then, label %else
then:
  br label %merge
else:
  br label %merge
merge:
  %blend = phi i32 [ 13, %then ], [ 12, %else ]
  %trunc = trunc i32 %blend to i8
  store i8 %trunc, ptr %arrayidx, align 1
  %rivMinus1 = add nuw nsw i32 %riv, -1
  %cond = icmp eq i32 %riv, 0
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}
