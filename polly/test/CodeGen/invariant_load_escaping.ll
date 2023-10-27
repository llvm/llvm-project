; RUN: opt %loadPolly -polly-codegen -polly-invariant-load-hoisting=true -S < %s | FileCheck %s
;
;    int f(int *A, int *B) {
;      // Possible aliasing between A and B but if not then *B would be
;      // invariant. We assume this and hoist *B but need to use a merged
;      // version in the return.
;      int i = 0;
;      int x = 0;
;
;      do {
;        x = *B;
;        A[i] += x;
;      } while (i++ < 100);
;
;      return x;
;    }
;
; CHECK: polly.preload.begin:
; CHECK:   %polly.access.B = getelementptr i32, ptr %B, i64 0
; CHECK:   %polly.access.B.load = load i32, ptr %polly.access.B
; CHECK:   store i32 %polly.access.B.load, ptr %tmp.preload.s2a
;
; CHECK: polly.merge_new_and_old:
; CHECK:   %tmp.merge = phi i32 [ %tmp.final_reload, %polly.exiting ], [ %tmp, %do.cond ]
; CHECK:   br label %do.end
;
; CHECK: do.end:
; CHECK:   ret i32 %tmp.merge
;
; CHECK: polly.loop_exit:
; CHECK:   %tmp.final_reload = load i32, ptr %tmp.preload.s2a
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @f(ptr %A, ptr %B) {
entry:
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %do.cond ], [ 0, %entry ]
  %tmp = load i32, ptr %B, align 4
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp1 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %tmp1, %tmp
  store i32 %add, ptr %arrayidx, align 4
  br label %do.cond

do.cond:                                          ; preds = %do.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 101
  br i1 %exitcond, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  ret i32 %tmp
}
