; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s

; void f(int a[], int N) {
;   int i;
;   for (i = 0; i < N; ++i)
;     a[i] = i;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(i64* nocapture %a, i64 %N) nounwind {
entry:
  br label %bb

bb:                                               ; preds = %bb, %entry
  %i = phi i64 [ 0, %entry ], [ %i.inc, %bb ]
  %scevgep = getelementptr inbounds i64, i64* %a, i64 %i
  store i64 %i, i64* %scevgep
  %i.inc = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.inc, %N
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}

; CHECK: Assumed Context:
; CHECK:   {  :  }
; CHECK: Arrays {
; CHECK:     i64 MemRef_a[*]; // Element size 8
; CHECK: }

; CHECK:  Stmt_bb
; CHECK:        Domain :=
; CHECK:            [N] -> { Stmt_bb[i0] : i0 >= 0 and i0 <= -1 + N };
; CHECK:        Schedule :=
; CHECK:            [N] -> { Stmt_bb[i0] -> [i0] };
; CHECK:        MustWriteAccess := [Reduction Type: NONE]
; CHECK:            [N] -> { Stmt_bb[i0] -> MemRef_a[i0] };
