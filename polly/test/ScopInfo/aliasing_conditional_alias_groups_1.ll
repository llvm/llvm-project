; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; Check that there is no alias group because we either access A or B never both.
;
; CHECK: Alias Groups (0):
;
;    void jd(int b, int *A, int *B) {
;      for (int i = 0; i < 1024; i++) {
;        if (b)
;          A[i] = A[i - 1];
;        else
;          B[i] = B[i - 1];
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32 %b, i32* %A, i32* %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %for.body
  %tmp = add nsw i64 %indvars.iv, -1
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %tmp
  %tmp3 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %tmp3, i32* %arrayidx2, align 4
  br label %if.end

if.else:                                          ; preds = %for.body
  %tmp4 = add nsw i64 %indvars.iv, -1
  %arrayidx5 = getelementptr inbounds i32, i32* %B, i64 %tmp4
  %tmp5 = load i32, i32* %arrayidx5, align 4
  %arrayidx7 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  store i32 %tmp5, i32* %arrayidx7, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
