; RUN: opt %loadPolly -basicaa -polly-scops -analyze < %s | FileCheck %s

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

; CHECK:      Invariant Accesses: {
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_for_j[i0, i1] -> MemRef_init_ptr[0] };
; CHECK-NEXT:         Execution Context: [N] -> {  : N <= -1 or N >= 1 }
; CHECK-NEXT: }
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_j
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N] -> { Stmt_for_j[i0, i1] : i0 >= 0 and i0 <= -1 + N and i1 >= 0 and i1 <= -1 + N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N] -> { Stmt_for_j[i0, i1] -> [i0, i1] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [N] -> { Stmt_for_j[i0, i1] -> MemRef_init[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N] -> { Stmt_for_j[i0, i1] -> MemRef_A[i1] };
; CHECK-NEXT: }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(i64* noalias %A, i64 %N, i64* noalias %init_ptr) #0 {
entry:
  br label %for.i

for.i:                                            ; preds = %for.i.end, %entry
  %indvar.i = phi i64 [ 0, %entry ], [ %indvar.i.next, %for.i.end ]
  %indvar.i.next = add nsw i64 %indvar.i, 1
  br label %entry.next

entry.next:                                       ; preds = %for.i
  %init = load i64, i64* %init_ptr
  br label %for.j

for.j:                                            ; preds = %for.j, %entry.next
  %indvar.j = phi i64 [ 0, %entry.next ], [ %indvar.j.next, %for.j ]
  %init_2 = load i64, i64* %init_ptr
  %init_sum = add i64 %init, %init_2
  %scevgep = getelementptr i64, i64* %A, i64 %indvar.j
  store i64 %init_sum, i64* %scevgep
  %indvar.j.next = add nsw i64 %indvar.j, 1
  %exitcond.j = icmp eq i64 %indvar.j.next, %N
  br i1 %exitcond.j, label %for.i.end, label %for.j

for.i.end:                                        ; preds = %for.j
  %exitcond.i = icmp eq i64 %indvar.i.next, %N
  br i1 %exitcond.i, label %return, label %for.i

return:                                           ; preds = %for.i.end
  ret void
}

attributes #0 = { nounwind }
