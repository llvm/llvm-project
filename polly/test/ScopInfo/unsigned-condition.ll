; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s

; void f(int a[], int N, unsigned P) {
;   int i;
;   for (i = 0; i < N; ++i)
;     if (P > 42)
;       a[i] = i;
; }

; The assumed context is the universe because the "signed-unsigned assumption"
;   [P, N] -> {  : N > 0 and P >= 0 }
; is implied by the execution domain. Thus if something is executed this
; assumption will hold.

; CHECK:      Assumed Context:
; CHECK-NEXT: [P, N] -> {  :  }
;
; CHECK:              Domain :=
; CHECK-NEXT:             [P, N] -> { Stmt_store[i0] : P >= 42 and 0 <= i0 < N };

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(i64* nocapture %a, i64 %N, i64 %P) nounwind {
entry:
  br label %bb

bb:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %bb.backedge ]
  %brcond = icmp uge i64 %P, 42
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
