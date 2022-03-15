; RUN: opt %loadPolly -polly-invariant-load-hoisting=true -polly-print-scops -disable-output < %s | FileCheck %s
;
;    void f(unsigned *restrict I, unsigned *restrict A, unsigned N, unsigned M) {
;      for (unsigned i = 0; i < N; i++) {
;        unsigned char V = *I;
;        if (V < M)
;          A[i]++;
;      }
;    }
;
; FIXME: The truncated value should be a paramter.
; CHECK:         Assumed Context:
; CHECK-NEXT:    [N, tmp, M] -> { : }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [N, tmp, M] -> { : N < 0 or (N > 0 and tmp >= 128) or (N > 0 and tmp < 0) or (N > 0 and M < 0) }
;
; CHECK:         Domain :=
; CHECK-NEXT:    [N, tmp, M] -> { Stmt_if_then[i0] : tmp >= 0 and M > tmp and 0 <= i0 < N };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* noalias %I, i32* noalias %A, i32 %N, i32 %M) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp = load i32, i32* %I, align 4
  %conv1 = and i32 %tmp, 255
  %cmp2 = icmp ult i32 %conv1, %M
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp1 = load i32, i32* %arrayidx, align 4
  %inc = add i32 %tmp1, 1
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
