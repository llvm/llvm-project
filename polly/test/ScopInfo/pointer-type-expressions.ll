; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s

; void f(int a[], int N, float *P) {
;   int i;
;   for (i = 0; i < N; ++i)
;     if (*P != 0)
;       a[i] = i;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(i64* nocapture %a, i64 %N, float * %P) nounwind {
entry:
  br label %bb

bb:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %bb.backedge ]
  %brcond = icmp ne float* %P, null
  br i1 %brcond, label %store, label %bb.backedge

store:
  %scevgep = getelementptr inbounds i64, i64* %a, i64 %i
  store i64 %i, i64* %scevgep
  br label %bb.backedge

bb.backedge:
  %i.inc = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.inc, %N
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

; CHECK:      Assumed Context:
; CHECK-NEXT:   {  :  }
; CHECK-NEXT: Invalid Context:
; CHECK-NEXT:   {  :  1 = 0 }

; CHECK:  Stmt_store
; CHECK:        Domain :=
; CHECK:            [P, N] -> { Stmt_store[i0] : 0 <= i0 < N and (P < 0 or P > 0) };
; CHECK:        Schedule :=
; CHECK:            [P, N] -> { Stmt_store[i0] -> [i0] : P < 0 or P > 0 };
; CHECK:        MustWriteAccess := [Reduction Type: NONE]
; CHECK:            [P, N] -> { Stmt_store[i0] -> MemRef_a[i0] };
