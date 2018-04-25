; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; The SCoP contains a loop with multiple exit blocks (BBs after leaving
; the loop). The current implementation of deriving their domain derives
; only a common domain for all of the exit blocks. We disabled loops with
; multiple exit blocks until this is fixed.
; XFAIL: *
;
; CHECK: Domain :=
; CHECK:   { Stmt_if_end[i0] : 0 <= i0 <= 1024 };
;
;    void f(int *A) {
;      int i = 0;
;      do {
;        if (i > 1024)
;          break;
;        A[i] = i;
;        i++;
;      } while (i > 0);
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A) {
entry:
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %do.cond ], [ 0, %entry ]
  %cmp = icmp sgt i64 %indvars.iv, 1024
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %do.body
  br label %do.end

if.end:                                           ; preds = %do.body
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp = trunc i64 %indvars.iv to i32
  store i32 %tmp, i32* %arrayidx, align 4
  br label %do.cond

do.cond:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp2 = icmp sgt i64 %indvars.iv.next, 0
  br i1 %cmp2, label %do.body, label %do.end.loopexit

do.end.loopexit:                                  ; preds = %do.cond
  br label %do.end

do.end:                                           ; preds = %do.end.loopexit, %if.then
  ret void
}
