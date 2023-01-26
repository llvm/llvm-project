; RUN: opt -opaque-pointers=0 %loadPolly \
; RUN: -S -polly-codegen < %s | FileCheck %s
;
;    void f(int *A, int N) {
;      for (int i = 0; i < N; i++)
;        if (A[i])
;          switch (i % 4) {
;          case 0:
;            A[i] += 1;
;            break;
;          case 1:
;            A[i] += 2;
;            break;
;          }
;    }
;
; CHECK: polly.stmt.if.then:
; CHECK:   %1 = trunc i64 %polly.indvar to i32
; CHECK:   %p_rem = srem i32 %1, 4
; CHECK:   switch i32 %p_rem, label %polly.stmt.sw.epilog [
; CHECK:     i32 0, label %polly.stmt.sw.bb
; CHECK:     i32 1, label %polly.stmt.sw.bb.3
; CHECK:   ]
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32 %N) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp1 = load i32, i32* %arrayidx, align 4
  %tobool = icmp eq i32 %tmp1, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  %tmp2 = trunc i64 %indvars.iv to i32
  %rem = srem i32 %tmp2, 4
  switch i32 %rem, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb.3
  ]

sw.bb:                                            ; preds = %if.then
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp3 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %tmp3, 1
  store i32 %add, i32* %arrayidx2, align 4
  br label %sw.epilog

sw.bb.3:                                          ; preds = %if.then
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp4 = load i32, i32* %arrayidx5, align 4
  %add6 = add nsw i32 %tmp4, 2
  store i32 %add6, i32* %arrayidx5, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb.3, %sw.bb, %if.then
  br label %if.end

if.end:                                           ; preds = %for.body, %sw.epilog
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
