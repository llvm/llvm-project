; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s

; void f(long A[], long N) {
;   long i;
;
;  if (true){
;    i = 0;
;    goto next;
;  }else{
;    i = 1;
;    goto next;
; }
;
; next:
;  if (true)
;    goto for.i;
;  else
;    goto for.i;
;
; for.i:
;   for (i = 0; i < N; ++i)
;     A[i] = i;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(ptr %A, i64 %N) nounwind {
entry:
  fence seq_cst
  br i1 true, label %then1, label %else1

then1:
  br label %next

else1:
  br label %next

next:
  br i1 true, label %then, label %else

then:
  br label %for.i.head

else:
  br label %for.i.head

for.i.head:
  br label %for.i.head1

for.i.head1:
  br label %for.i

for.i:
  %indvar = phi i64 [ 0, %for.i.head1], [ %indvar.next, %for.i ]
  fence seq_cst
  %scevgep = getelementptr i64, ptr %A, i64 %indvar
  store i64 %indvar, ptr %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %return, label %for.i

return:
  fence seq_cst
  ret void
}

; CHECK: Valid Region for Scop: next => for.i
