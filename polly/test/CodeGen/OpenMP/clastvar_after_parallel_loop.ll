; RUN: opt %loadPolly -basicaa -polly-cloog -analyze -S < %s | FileCheck %s -check-prefix=CLOOG
; RUN: opt %loadPolly -basicaa -polly-codegen -enable-polly-openmp -S < %s | FileCheck %s
;
; Test case that checks that after the parallel loop on j the value for i is
; taken from the right temporary (in particular, _not_ the temporary used for i
; in the OpenMP subfunction for the loop on j).
;
; void f(long * restrict A) {
;     long i, j;
;     for (i=0; i<100; ++i) {
;         #pragma omp parallel
;         for (j=0; j<100; ++j)
;             A[j] += i;
;         A[i] = 42;
;     }
; }
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i64* noalias nocapture %A) {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ %i.next, %for.end ], [ 0, %entry ]
  br label %for.j

for.j:                                     ; preds = %for.j, %for.i
  %j = phi i64 [ 0, %for.i ], [ %j.next, %for.j ]
  %i.arrayidx = getelementptr inbounds i64* %A, i64 %j
  %load = load i64* %i.arrayidx
  %add = add nsw i64 %load, %i
  store i64 %add, i64* %i.arrayidx
  %j.next = add i64 %j, 1
  %j.exitcond = icmp eq i64 %j.next, 100
  br i1 %j.exitcond, label %for.end, label %for.j

for.end:                                       ; preds = %for.j
  %j.arrayidx = getelementptr inbounds i64* %A, i64 %i
  store i64 42, i64* %j.arrayidx
  %i.next = add i64 %i, 1
  %i.exitcond = icmp eq i64 %i.next, 100
  br i1 %i.exitcond, label %end, label %for.i

end:                                         ; preds = %for.end, %entry
  ret void
}

; CLOOG: for (c2=0;c2<=99;c2++) {
; CLOOG:   for (c4=0;c4<=99;c4++) {
; CLOOG:     Stmt_for_j(c2,c4);
; CLOOG:   }
; CLOOG:   Stmt_for_end(c2);
; CLOOG: }

; CHECK: @f.omp_subfn
