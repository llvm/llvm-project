; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; Check that we create two alias groups since the minimal/maximal accesses
; depend on %b.
;
; CHECK: Alias Groups (2):
;
;    void jd(int b, int *A, int *B) {
;      for (int i = 0; i < 1024; i++) {
;        if (b)
;          A[i] = B[5];
;        else
;          B[i] = A[7];
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32 %b, ptr %A, ptr %B) {
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
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 5
  %tmp = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %tmp, ptr %arrayidx1, align 4
  br label %if.end

if.else:                                          ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 7
  %tmp1 = load i32, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  store i32 %tmp1, ptr %arrayidx4, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
