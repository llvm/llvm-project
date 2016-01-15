; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s

; void f(long a[][128], long N, long M) {
;   long i, j;
;   for (j = 0; j < (4*N + 7*M +3); ++j)
;     for (i = (7*j + 6*M -9); i < (3*j + 5*N + 2) ; ++i)
;         a[i][j] = 0;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f([128 x i64]* nocapture %a, i64 %N, i64 %M) nounwind {
entry:
  %0 = shl i64 %N, 2
  %1 = mul i64 %M, 7
  %2 = or i64 %0, 3
  %3 = add nsw i64 %2, %1
  %4 = icmp sgt i64 %3, 0
  br i1 %4, label %bb.nph8, label %return

bb.nph8:                                          ; preds = %entry
  %tmp14 = mul i64 %M, 6
  %tmp15 = add i64 %tmp14, -9
  %tmp20 = add i64 %1, %0
  %tmp21 = add i64 %tmp20, 3
  %tmp25 = mul i64 %M, -6
  %tmp26 = mul i64 %N, 5
  %tmp27 = add i64 %tmp25, %tmp26
  %tmp28 = add i64 %tmp27, 11
  %tmp35 = add i64 %tmp26, 2
  br label %bb

bb:                                               ; preds = %bb3, %bb.nph8
  %j.07 = phi i64 [ 0, %bb.nph8 ], [ %6, %bb3 ]
  %tmp17 = mul i64 %j.07, 897
  %tmp24 = mul i64 %j.07, -4
  %tmp13 = add i64 %tmp24, %tmp28
  %tmp30 = mul i64 %j.07, 7
  %tmp33 = add i64 %tmp30, %tmp15
  %tmp34 = mul i64 %j.07, 3
  %tmp36 = add i64 %tmp34, %tmp35
  %5 = icmp sgt i64 %tmp36, %tmp33
  br i1 %5, label %bb1, label %bb3

bb1:                                              ; preds = %bb1, %bb
  %indvar = phi i64 [ 0, %bb ], [ %indvar.next, %bb1 ]
  %tmp16 = add i64 %indvar, %tmp15
  %scevgep = getelementptr [128 x i64], [128 x i64]* %a, i64 %tmp16, i64 %tmp17
  store i64 0, i64* %scevgep
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %tmp13
  br i1 %exitcond, label %bb3, label %bb1

bb3:                                              ; preds = %bb1, %bb
  %6 = add nsw i64 %j.07, 1
  %exitcond22 = icmp eq i64 %6, %tmp21
  br i1 %exitcond22, label %return, label %bb

return:                                           ; preds = %bb3, %entry
  ret void
}


; CHECK:      p0: %N
; CHECK-NEXT: p1: %M
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_bb1
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N, M] -> { Stmt_bb1[i0, i1] : i0 >= 0 and i0 <= 2 + 4N + 7M and i1 <= 10 + 5N - 6M - 4i0 and i1 >= 0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N, M] -> { Stmt_bb1[i0, i1] -> [i0, i1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N, M] -> { Stmt_bb1[i0, i1] -> MemRef_a[-9 + 6M + i1, 897i0] };
; CHECK-NEXT: }
