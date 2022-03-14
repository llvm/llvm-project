; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s

; void f(long A[], long N) {
;   long i;
;
;  if (true)
;    goto loop;
;  else
;    goto loop;
;
; loop:
;   for (i = 0; i < N; ++i)
;     A[i] = i;
; }

; We will detect this scop even if the loop is not in lcssa form

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(i64* %A, i64 %N) nounwind {
entry:
  fence seq_cst
  br i1 true, label %then, label %else

then:
  br label %for.i

else:
  br label %for.i

for.i:
  %indvar = phi i64 [ 0, %then ], [ 0, %else], [ %indvar.next, %for.i ]
  %scevgep = getelementptr i64, i64* %A, i64 %indvar
  store i64 %indvar, i64* %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %return, label %for.i

return:
  fence seq_cst
  ret void
}

; CHECK: Valid Region for Scop
