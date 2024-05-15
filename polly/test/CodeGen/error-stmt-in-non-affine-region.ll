; RUN: opt %loadPolly -S -polly-codegen < %s | FileCheck %s
; XFAIL: *
;
; CHECK-LABEL: polly.stmt.if.then:
; CHECK-NEXT:   unreachable
;
;    void f(int *A, int N) {
;      for (int i = 0; i < 1024; i++)
;        if (i == N) {
;          if (A[i])
;            abort();
;          else
;            A[i] = i;
;        }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, i64 %N) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp = load i32, ptr %arrayidx, align 4
  %cmp.outer = icmp eq i64 %indvars.iv, %N
  br i1 %cmp.outer, label %if.then.outer, label %for.inc

if.then.outer:
  %tobool = icmp eq i32 %tmp, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %for.body
  call void @abort()
  unreachable

if.else:                                          ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp1 = trunc i64 %indvars.iv to i32
  store i32 %tmp1, ptr %arrayidx2, align 4
  br label %if.end

if.end:                                           ; preds = %if.else
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare void @abort()
