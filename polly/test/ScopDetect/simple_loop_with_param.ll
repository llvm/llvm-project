; RUN: opt %loadPolly -basicaa -polly-detect -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -basicaa -polly-detect -analyze -polly-codegen-scev < %s | FileCheck -check-prefix=CHECK-SCEV %s

; void f(long A[], long N, long *init_ptr) {
;   long i, j;
;   long i_non_canonical;
;
;   for (i = 0; i < N; ++i) {
;     init = *init_ptr;
;     i_non_canonical = init;
;     for (i = 0; i < N; ++i) {
;       A[i] = i_non_canonical;
;       i_non_canonical += 1;
;     }
;   }
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i64* noalias %A, i64 %N, i64* noalias %init_ptr) nounwind {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ 0, %entry ], [ %indvar.i.next, %for.i.end ]
  %indvar.i.next = add nsw i64 %indvar.i, 1
  br label %entry.next

entry.next:
  %init = load i64* %init_ptr
  br label %for.j

for.j:
  %indvar.j = phi i64 [ 0, %entry.next ], [ %indvar.j.next, %for.j ]
  %indvar.j.non_canonical = phi i64 [ %init, %entry.next ], [ %indvar.j.non_canonical.next, %for.j ]
  %scevgep = getelementptr i64* %A, i64 %indvar.j
  store i64 %indvar.j.non_canonical, i64* %scevgep
  %indvar.j.next = add nsw i64 %indvar.j, 1
  %indvar.j.non_canonical.next = add nsw i64 %indvar.j.non_canonical, 1
  %exitcond.j = icmp eq i64 %indvar.j.next, %N
  br i1 %exitcond.j, label %for.i.end, label %for.j

for.i.end:
  %exitcond.i = icmp eq i64 %indvar.i.next, %N
  br i1 %exitcond.i, label %return, label %for.i

return:
  ret void
}

; CHECK-NOT: Valid Region for Scop
; CHECK-SCEV: Valid Region for Scop: for.j => for.i.end
