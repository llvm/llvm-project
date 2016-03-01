; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; Allow the user to define function names that are treated as
; error functions and assumed not to be executed.
;
;    void timer_start(void);
;    void timer_stop(void);
;    void kernel(int *A, int *B, int timeit, int N) {
;
;      if (timeit)
;        timer_start();
;
;      for (int i = 0; i < N; i++)
;        A[i] += B[i];
;
;      if (timeit) {
;        timer_stop();
;        timer_start();
;      }
;
;      for (int i = 0; i < N; i++)
;        A[i] += B[i];
;
;      if (timeit)
;        timer_stop();
;    }
;
; CHECK:      Region: %for.cond---%if.end.20
; CHECK:      Assumed Context:
; CHECK-NEXT: [N, timeit] -> {  :  }
; CHECK:      Invalid Context:
; CHECK-NEXT: [N, timeit] -> {  : timeit < 0 or timeit > 0 }
; CHECK:      Statements {
; CHECK:        Stmt
; CHECK:        Stmt
; CHECK-NOT:    Stmt
; CHECK:      }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @kernel(i32* %A, i32* %B, i32 %timeit, i32 %N) {
entry:
  %tobool = icmp eq i32 %timeit, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  call void @timer_start()
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.end
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.inc ], [ 0, %if.end ]
  %cmp = icmp slt i64 %indvars.iv1, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv1
  %tmp3 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv1
  %tmp4 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %tmp4, %tmp3
  store i32 %add, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %tobool3 = icmp eq i32 %timeit, 0
  br i1 %tobool3, label %if.end.5, label %if.then.4

if.then.4:                                        ; preds = %for.end
  call void @timer_stop()
  call void @timer_start()
  br label %if.end.5

if.end.5:                                         ; preds = %for.end, %if.then.4
  %tmp5 = sext i32 %N to i64
  br label %for.cond.7

for.cond.7:                                       ; preds = %for.inc.15, %if.end.5
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc.15 ], [ 0, %if.end.5 ]
  %cmp8 = icmp slt i64 %indvars.iv, %tmp5
  br i1 %cmp8, label %for.body.9, label %for.end.17

for.body.9:                                       ; preds = %for.cond.7
  %arrayidx11 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %tmp6 = load i32, i32* %arrayidx11, align 4
  %arrayidx13 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp7 = load i32, i32* %arrayidx13, align 4
  %add14 = add nsw i32 %tmp7, %tmp6
  store i32 %add14, i32* %arrayidx13, align 4
  br label %for.inc.15

for.inc.15:                                       ; preds = %for.body.9
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond.7

for.end.17:                                       ; preds = %for.cond.7
  %tobool18 = icmp eq i32 %timeit, 0
  br i1 %tobool18, label %if.end.20, label %if.then.19

if.then.19:                                       ; preds = %for.end.17
  call void @timer_stop()
  br label %if.end.20

if.end.20:                                        ; preds = %for.end.17, %if.then.19
  ret void
}

declare void @timer_start()
declare void @timer_stop()
