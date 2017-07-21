; RUN: opt %loadPolly -polly-scops -analyze \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s
;
;    void f(unsigned long *restrict I, unsigned *restrict A, unsigned N) {
;      for (unsigned i = 0; i < N; i++) {
;        unsigned V = *I;
;        if (V < i)
;          A[i]++;
;      }
;    }
;
; CHECK:         Assumed Context:
; CHECK-NEXT:    [N, tmp] -> { : }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [N, tmp] -> { : N > 0 and (tmp < 0 or tmp >= 2147483648) }
;
; CHECK:         Domain :=
; CHECK-NEXT:    [N, tmp] -> { Stmt_if_then[i0] : tmp >= 0 and tmp < i0 < N };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i64* noalias %I, i32* noalias %A, i32 %N, i32 %M) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp = load i64, i64* %I, align 8
  %conv = trunc i64 %tmp to i32
  %tmp1 = zext i32 %conv to i64
  %cmp1 = icmp ult i64 %tmp1, %indvars.iv
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp2 = load i32, i32* %arrayidx, align 4
  %inc = add i32 %tmp2, 1
  store i32 %inc, i32* %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
