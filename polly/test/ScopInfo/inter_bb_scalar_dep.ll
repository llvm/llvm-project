; RUN: opt %loadPolly -basic-aa -polly-print-scops \
; RUN: -polly-invariant-load-hoisting=true -disable-output < %s | FileCheck %s

; void f(long A[], int N, int *init_ptr) {
;   long i, j;
;
;   for (i = 0; i < N; ++i) {
;     init = *init_ptr;
;     for (i = 0; i < N; ++i) {
;       A[i] = init + 2;
;     }
;   }
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

; Function Attrs: nounwind
; CHECK: Invariant
; CHECK:           ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:          MemRef_init_ptr[0]

define void @f(ptr noalias %A, i64 %N, ptr noalias %init_ptr) #0 {
entry:
  br label %for.i

for.i:                                            ; preds = %for.i.end, %entry
  %indvar.i = phi i64 [ 0, %entry ], [ %indvar.i.next, %for.i.end ]
  %indvar.i.next = add nsw i64 %indvar.i, 1
  br label %entry.next

entry.next:                                       ; preds = %for.i
  %init = load i64, ptr %init_ptr
; CHECK-NOT: Stmt_entry_next
  br label %for.j

for.j:                                            ; preds = %for.j, %entry.next
  %indvar.j = phi i64 [ 0, %entry.next ], [ %indvar.j.next, %for.j ]
  %init_plus_two = add i64 %init, 2
; CHECK-LABEL: Stmt_for_j
; CHECK:           ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:          [N] -> { Stmt_for_j[i0, i1] -> MemRef_init[] };
; CHECK:           MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:          [N] -> { Stmt_for_j[i0, i1] -> MemRef_A[i1] };
  %scevgep = getelementptr i64, ptr %A, i64 %indvar.j
  store i64 %init_plus_two, ptr %scevgep
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
