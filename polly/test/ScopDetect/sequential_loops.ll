; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

; Two sequential loops right after each other.
;
; void f(long A[], long N) {
;   long i;
;   for (i = 0; i < N; ++i)
;     A[i] = i;
;   for (i = 0; i < N; ++i)
;     A[i] = i;
; }

define void @f1(i64* %A, i64 %N) nounwind {
; CHECK-LABEL: 'Polly - Detect static control parts (SCoPs)' for function 'f1'
entry:
  fence seq_cst
  br label %for.i.1

for.i.1:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.i.1 ]
  %scevgep = getelementptr i64, i64* %A, i64 %indvar
  store i64 %indvar, i64* %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %for.i.2, label %for.i.1

for.i.2:
  %indvar.2 = phi i64 [ 0, %for.i.1 ], [ %indvar.next.2, %for.i.2 ]
  %scevgep.2 = getelementptr i64, i64* %A, i64 %indvar.2
  store i64 %indvar.2, i64* %scevgep.2
  %indvar.next.2 = add nsw i64 %indvar.2, 1
  %exitcond.2 = icmp eq i64 %indvar.next.2, %N
  br i1 %exitcond.2, label %return, label %for.i.2

return:
  fence seq_cst
  ret void
}

; C-H-E-C-K: Valid Region for Scop: for.i.1 => return
; This one is currently not completely detected due to the PHI node in
; for.i.2 causing a 'PHI node in exit BB' error for the first loop. This should
; be fixed at some point. Such test cases do not really show up for us, as
; the -loop-simplify pass always inserts a preheader as in the test case below.

; Two sequential loops with a basic block in between.
;
;     void f(long A[], long N) {
;       long i;
;
;       for (i = 0; i < N; ++i)
;         A[i] = i;
; preheader:
;       ;
;
;       for (i = 0; i < N; ++i)
;         A[i] = i;
;     }

define void @f2(i64* %A, i64 %N) nounwind {
; CHECK-LABEL: 'Polly - Detect static control parts (SCoPs)' for function 'f2'
entry:
  fence seq_cst
  br label %for.i.1

for.i.1:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.i.1 ]
  %scevgep = getelementptr i64, i64* %A, i64 %indvar
  store i64 %indvar, i64* %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %preheader, label %for.i.1

preheader:
  br label %for.i.2

for.i.2:
  %indvar.2 = phi i64 [ 0, %preheader ], [ %indvar.next.2, %for.i.2 ]
  %scevgep.2 = getelementptr i64, i64* %A, i64 %indvar.2
  store i64 %indvar.2, i64* %scevgep.2
  %indvar.next.2 = add nsw i64 %indvar.2, 1
  %exitcond.2 = icmp eq i64 %indvar.next.2, %N
  br i1 %exitcond.2, label %return, label %for.i.2

return:
  fence seq_cst
  ret void
}

; CHECK: Valid Region for Scop: for.i.1 => return
