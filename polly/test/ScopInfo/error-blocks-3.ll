; RUN: opt %loadPolly -polly-print-scops -polly-detect-keep-going -polly-allow-nonaffine -disable-output < %s | FileCheck %s
;
; The instruction
;
;   %idxprom = sext i32 %call to i64
;
; uses an argument that is inside and error block. Since error blocks are
; removed from the SCoP, the argument is not available. Polly currently
; does not consider that %idxprom itself is an error block as well.
;
; This also tests that -polly-detect-keep-going still correctly rejects this SCoP.
; https://llvm.org/PR58484
;
; CHECK:      Printing analysis 'Polly - Create polyhedral description of Scops' for region: 'for.cond => for.end' in function 'g':
; CHECK-NEXT: Invalid Scop!
;
;    int f();
;    void g(int *A, int N) {
;      for (int i = 0; i < N; i++) {
;        if (i > 512) {
;          int v = f();
;        S:
;          A[v]++;
;        }
;        A[i]++;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @g(ptr %A, i32 %N) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %cmp1 = icmp sgt i64 %indvars.iv, 512
  br i1 %cmp1, label %if.then, label %if.end3

if.then:                                          ; preds = %for.body
  %call = call i32 (...) @f()
  br label %S

S:                                                ; preds = %if.then
  %idxprom = sext i32 %call to i64
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %idxprom
  %tmp1 = load i32, ptr %arrayidx, align 4
  %inc = add nsw i32 %tmp1, 1
  store i32 %inc, ptr %arrayidx, align 4
  br label %if.end3

if.end3:                                          ; preds = %if.end, %for.body
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp2 = load i32, ptr %arrayidx5, align 4
  %inc6 = add nsw i32 %tmp2, 1
  store i32 %inc6, ptr %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end3, %if.then2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare i32 @f(...)
