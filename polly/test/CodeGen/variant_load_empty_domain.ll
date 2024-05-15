; RUN: opt %loadPolly -polly-codegen -disable-output < %s
;
;
;    void f(int *A) {
;      for (int i = 1; i < 10; i++) {
;        A[i]++;
;        if (i > 10)
;          A[i] += A[0];
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 1, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 10
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp = load i32, ptr %arrayidx, align 4
  %inc = add nsw i32 %tmp, 1
  store i32 %inc, ptr %arrayidx, align 4
  br i1 false, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %tmp1 = load i32, ptr %A, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp2 = load i32, ptr %arrayidx4, align 4
  %add = add nsw i32 %tmp2, %tmp1
  store i32 %add, ptr %arrayidx4, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
