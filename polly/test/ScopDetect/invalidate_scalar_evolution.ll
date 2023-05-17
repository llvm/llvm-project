; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s -check-prefix=PHI

; void f(long A[], long N) {
;   long i;
;   for (i = 0; i < N; ++i)
;     A[i] = i;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(ptr %A, i64 %N, i64 %p) nounwind {
entry:
  fence seq_cst
  br label %pre

pre:
  %p_tmp = srem i64 %p, 5
  br i1 true, label %for.i, label %then

for.i:
  %indvar = phi i64 [ 0, %pre ], [ %indvar.next, %for.i ]
  %indvar.p1 = phi i64 [ 0, %pre ], [ %indvar.p1.next, %for.i ]
  %indvar.p2 = phi i64 [ 0, %pre ], [ %indvar.p2.next, %for.i ]
  %sum = add i64 %indvar, %indvar.p1
  %sum2 = sub i64 %sum, %indvar.p2
  %scevgep = getelementptr i64, ptr %A, i64 %indvar
  store i64 %indvar, ptr %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %indvar.p1.next = add nsw i64 %indvar.p1, %p_tmp
  %indvar.p2.next = add nsw i64 %indvar.p2, %p_tmp
  %exitcond = icmp eq i64 %sum2, %N
  br i1 %exitcond, label %then, label %for.i

then:
  br label %return

return:
  fence seq_cst
  ret void
}

; PHI: Valid Region for Scop: pre => return
