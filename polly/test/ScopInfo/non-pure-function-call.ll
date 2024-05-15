; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK:      Assumed Context:
; CHECK-NEXT:   [N] -> {  :  }
; CHECK:      Invalid Context:
; CHECK-NEXT:   [N] -> {  : N >= 102 }
;
;    void g(void);
;    void f(int *A, int N) {
;      for (int i = 0; i < N; i++) {
;        if (i > 100)
;          g();
;        A[i]++;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, i32 %N) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %cmp1 = icmp sgt i64 %indvars.iv, 100
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  call void @g() #2
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp1 = load i32, ptr %arrayidx, align 4
  %inc = add nsw i32 %tmp1, 1
  store i32 %inc, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare void @g()
