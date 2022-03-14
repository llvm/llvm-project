; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -disable-output < %s

; void f(long A[], long N) {
;   long i, j;
;   if (true)
;     for (j = 0; j < N; ++j)
;       for (i = 0; i < N; ++i)
;         A[i] = i;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(i64* %A, i64 %N) nounwind {
entry:
  fence seq_cst
  br label %next

next:
  br i1 true, label %for.j, label %return

for.j:
  %j.015 = phi i64 [ %inc5, %for.inc8 ], [ 0, %next ]
  br label %for.i

for.i:
  %indvar = phi i64 [ 0, %for.j], [ %indvar.next, %for.i ]
  %scevgep = getelementptr i64, i64* %A, i64 %indvar
  store i64 %indvar, i64* %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %for.inc8, label %for.i

for.inc8:                                         ; preds = %for.body3
  %inc5 = add nsw i64 %j.015, 1
  %exitcond16 = icmp eq i64 %inc5, %N
  br i1 %exitcond16, label %return, label %for.j

return:
  fence seq_cst
  ret void
}

; CHECK: Valid Region for Scop: next => return
