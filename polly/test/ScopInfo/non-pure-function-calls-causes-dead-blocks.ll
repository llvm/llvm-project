; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; Error blocks are skipped during SCoP detection. We skip them during
; SCoP formation too as they might contain instructions we can not handle.
; However statements / basic blocks that follow error blocks are modeled.
;
;    void timer_start(void);
;    void timer_stop(void);
;    void kernel(int *A, int *B, int timeit, int N) {
;
;      if (timeit) {
;        timer_start();
;        // split BB
;        A[0] = 0;                 // Do not create a statement for this block
;      }
;
;      for (int i = 0; i < N; i++)
;        A[i] += B[i];
;
;      if (timeit) {
;        timer_stop();
;        if (invalid float branch) // Do not crash on the float branch
;          timer_start();
;      }
;
;      for (int i = 0; i < N; i++)
;        A[i] += B[i];
;
;      if (timeit)
;        timer_stop();
;    }
;
;  The assumed context should not be empty even though all statements are
;  executed only if timeit != 0.
;
; CHECK:    Region: %entry.split---%if.end.20
; CHECK:    Assumed Context:
; CHECK-NEXT:    [timeit, N] -> { : }
; CHECK:    Invalid Context:
; CHECK-NEXT:    [timeit, N] -> { : timeit < 0 or timeit > 0 }
; CHECK:    Statements {
; CHECK-NOT:      Stmt_if_then_split
; CHECK:      Stmt_for_body
; CHECK:      Stmt_for_body_9
; CHECK:    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @kernel(ptr %A, ptr %B, i32 %timeit, i32 %N) {
entry:
  br label %entry.split

entry.split:
  %tobool = icmp eq i32 %timeit, 0
  br i1 %tobool, label %for.cond.pre, label %if.then

if.then:                                          ; preds = %entry
  call void @timer_start()
  br label %if.then.split

; Dead block if we assume if.then not to be executed because of the call
if.then.split:                                           ; preds = %if.then
  store i32 0, ptr %A, align 4
  br label %for.cond.pre

for.cond.pre:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.end
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.inc ], [ 0, %for.cond.pre ]
  %cmp = icmp slt i64 %indvars.iv1, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv1
  %tmp3 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv1
  %tmp4 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %tmp4, %tmp3
  store i32 %add, ptr %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %tobool3 = icmp eq i32 %timeit, 0
  br i1 %tobool3, label %if.end.5, label %if.then.4

if.then.4:                                        ; preds = %for.end
  call void @timer_stop()
  %na = fcmp one float 4.0, 5.0
  br i1 %na, label %if.end.5, label %if.then.4.rem

if.then.4.rem:                                        ; preds = %for.end
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
  %arrayidx11 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %tmp6 = load i32, ptr %arrayidx11, align 4
  %arrayidx13 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp7 = load i32, ptr %arrayidx13, align 4
  %add14 = add nsw i32 %tmp7, %tmp6
  store i32 %add14, ptr %arrayidx13, align 4
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
