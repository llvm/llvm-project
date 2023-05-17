; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s

; void f(long A[], long N) {
;   long i;
;   if (true)
;     for (i = 0; i < N; ++i)
;       A[i] = i;
;   else
;     for (j = 0; j < N; ++j)
;        A[j] = j;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(ptr %A, i64 %N) nounwind {
entry:
  fence seq_cst
  br i1 true, label %next, label %next2

next2:
  br i1 true, label %for.j, label %return

for.j:
  %indvar2 = phi i64 [ 0, %next2], [ %indvar2.next2, %for.j]
  %scevgep2 = getelementptr i64, ptr %A, i64 %indvar2
  store i64 %indvar2, ptr %scevgep2
  %indvar2.next2 = add nsw i64 %indvar2, 1
  %exitcond2 = icmp eq i64 %indvar2.next2, %N
  br i1 %exitcond2, label %return, label %for.j

next:
  br i1 true, label %for.i, label %return

for.i:
  %indvar = phi i64 [ 0, %next], [ %indvar.next, %for.i ]
  %scevgep = getelementptr i64, ptr %A, i64 %indvar
  store i64 %indvar, ptr %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %return, label %for.i

return:
  fence seq_cst
  ret void
}

; CHECK: Valid Region for Scop: next => return
; CHECK: Valid Region for Scop: next2 => return
