; RUN: opt %loadPolly -polly-print-scops -polly-invariant-load-hoisting=true -disable-output < %s | FileCheck %s
;
; Check that we do not try to preload *I and assume p != 42.
;
; CHECK:      Invariant Accesses: {
; CHECK-NEXT: }
;
; CHECK:      Invalid Context:
; CHECK-NEXT: [N, p] -> {  : false }
;
; CHECK:      Stmt_if_then__TO__if_end
; CHECK-NEXT:   Domain :=
; CHECK-NEXT:   [N, p] -> { Stmt_if_then__TO__if_end[i0] : p = 42 and 0 <= i0 < N };
;
;    void f(int *A, int *I, int N, int p, int q) {
;      for (int i = 0; i < N; i++) {
;        if (p == 42) {
;          *I = 0;
;          if (*I == q)
;            A[i] *= 2;
;        }
;        A[i]++;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, ptr %I, i32 %N, i32 %p, i32 %q) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %cmp1 = icmp eq i32 %p, 42
  br i1 %cmp1, label %if.then, label %if.end4

if.then:                                          ; preds = %for.body
  store i32 0, ptr %I, align 4
  %tmp1 = load i32, ptr %I, align 4
  %cmp2 = icmp eq i32 %tmp1, %q
  br i1 %cmp2, label %if.then3, label %if.end

if.then3:                                         ; preds = %if.then
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp2 = load i32, ptr %arrayidx, align 4
  %mul = shl nsw i32 %tmp2, 1
  store i32 %mul, ptr %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %if.then3, %if.then
  br label %if.end4

if.end4:                                          ; preds = %if.end, %for.body
  %arrayidx6 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp3 = load i32, ptr %arrayidx6, align 4
  %inc = add nsw i32 %tmp3, 1
  store i32 %inc, ptr %arrayidx6, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
