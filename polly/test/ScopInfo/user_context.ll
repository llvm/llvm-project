; RUN: opt %loadPolly                                      -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-context='[N] -> {: N = 1024}' -polly-print-scops -disable-output < %s | FileCheck %s --check-prefix=CTX
; RUN: opt %loadPolly -polly-context='[N,M] -> {: 1 = 0}'  -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-context='[] -> {: 1 = 0}'     -polly-print-scops -disable-output < %s | FileCheck %s

; void f(int a[], int N) {
;   int i;
;   for (i = 0; i < N; ++i)
;     a[i] = i;
; }

; CHECK:      Context:
; CHECK-NEXT: [N] -> {  : -9223372036854775808 <= N <= 9223372036854775807 }

; CTX:      Context:
; CTX-NEXT: [N] -> {  : N = 1024 }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(ptr nocapture %a, i64 %N) nounwind {
entry:
  br label %bb

bb:                                               ; preds = %bb, %entry
  %i = phi i64 [ 0, %entry ], [ %i.inc, %bb ]
  %scevgep = getelementptr i64, ptr %a, i64 %i
  store i64 %i, ptr %scevgep
  %i.inc = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.inc, %N
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}

