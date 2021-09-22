; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; Test case to trigger the hard way of creating a unique entering
; edge for the SCoP. It is triggered because the entering edge
; here: %while.begin --> %if is __not__ critical.
;
;    int f(void);
;    void jd(int b, int *A) {
;      while (f()) {
;        if (b)
;          for (int i = 0; i < 1024; i++)
;            A[i] = i;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32 %b, i32* %A) {
entry:
  br label %while.begin

; CHECK-LABEL: while.begin.region_exiting:
; CHECK:         br label %polly.merge_new_and_old

; CHECK-LABEL: while.begin:
while.begin:
; CHECK:  %call = call i32 @f()
  %call = call i32 @f()
; CHECK:  %tobool = icmp eq i32 %call, 0
  %tobool = icmp eq i32 %call, 0
; CHECK:  br i1 %tobool, label %while.end, label %polly.split_new_and_old
  br i1 %tobool, label %while.end, label %if

; CHECK: polly.split_new_and_old:
; CHECK:   br i1 true, label %polly.start, label %if

; CHECK: if:
if:                                               ; preds = %while.begin
; CHECK: %tobool2 = icmp eq i32 %b, 0
  %tobool2 = icmp eq i32 %b, 0
; CHECK: br i1 %tobool2, label %while.begin{{[a-zA-Z._]*}}, label %for.cond
  br i1 %tobool2, label %while.begin, label %for.cond

for.cond:                                         ; preds = %for.inc, %if
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %if ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %while.begin

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp = trunc i64 %indvars.iv to i32
  store i32 %tmp, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

while.end:                                        ; preds = %entry, %for.cond
  ret void
}

declare i32 @f()
