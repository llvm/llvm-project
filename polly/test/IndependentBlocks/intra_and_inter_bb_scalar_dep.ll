; RUN: opt %loadPolly -basicaa -polly-independent -S < %s | FileCheck %s
; RUN: opt %loadPolly -basicaa -polly-independent -polly-codegen-scev -S < %s | FileCheck %s

; void f(long A[], int N, int *init_ptr) {
;   long i, j;
;
;   for (i = 0; i < N; ++i) {
;     init = *init_ptr;
;     for (i = 0; i < N; ++i) {
;       init2 = *init_ptr;
;       A[i] = init + init2;
;     }
;   }
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i64* noalias %A, i64 %N, i64* noalias %init_ptr) nounwind {
entry:

; CHECK: entry
; CHECK: %init.s2a = alloca i64
; CHECK: br label %for.i
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
  %init_2 = load i64* %init_ptr
  %init_sum = add i64 %init, %init_2
; CHECK: %init_2 = load i64* %init_ptr
; CHECK: %init.loadarray = load i64* %init.s2a
; CHECK: %init_sum = add i64 %init.loadarray, %init_2

; The SCEV of %init_sum is (%init + %init_2). It is referring to both an
; UnknownValue in the same and in a different basic block. We want only the
; reference to the different basic block to be replaced.
  %scevgep = getelementptr i64* %A, i64 %indvar.j
  store i64 %init_sum, i64* %scevgep
  %indvar.j.next = add nsw i64 %indvar.j, 1
  %exitcond.j = icmp eq i64 %indvar.j.next, %N
  br i1 %exitcond.j, label %for.i.end, label %for.j

for.i.end:
  %exitcond.i = icmp eq i64 %indvar.i.next, %N
  br i1 %exitcond.i, label %return, label %for.i

return:
  ret void
}
