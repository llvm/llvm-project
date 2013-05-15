; RUN: opt %loadPolly -basicaa -polly-independent -S < %s | FileCheck %s
; RUN: opt %loadPolly -basicaa -polly-independent -polly-codegen-scev -S < %s | FileCheck %s

; void f(long A[], int N, int *init_ptr) {
;   long i, j;
;
;   for (i = 0; i < N; ++i) {
;     for (i = 0; i < N; ++i) {
;       init = *init_ptr;
;       A[i] = init + 2;
;     }
;   }
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i64* noalias %A, i64 %N, i64* noalias %init_ptr) nounwind {
entry:

; CHECK: entry
; CHECK: br label %for.i
  br label %for.i

for.i:
  %indvar.i = phi i64 [ 0, %entry ], [ %indvar.i.next, %for.i.end ]
  %indvar.i.next = add nsw i64 %indvar.i, 1
  br label %entry.next

entry.next:
  br label %for.j

for.j:
  %indvar.j = phi i64 [ 0, %entry.next ], [ %indvar.j.next, %for.j ]
  %init = load i64* %init_ptr
  %init_plus_two = add i64 %init, 2
; The scalar evolution of %init_plus_two is (2 + %init). So we have a
; non-trivial scalar evolution referring to a value in the same basic block.
; We want to ensure that this scalar is not translated into a memory copy.
  %scevgep = getelementptr i64* %A, i64 %indvar.j
  store i64 %init_plus_two, i64* %scevgep
  %indvar.j.next = add nsw i64 %indvar.j, 1
  %exitcond.j = icmp eq i64 %indvar.j.next, %N
  br i1 %exitcond.j, label %for.i.end, label %for.j

for.i.end:
  %exitcond.i = icmp eq i64 %indvar.i.next, %N
  br i1 %exitcond.i, label %return, label %for.i

return:
  ret void
}
