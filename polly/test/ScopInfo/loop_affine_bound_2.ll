; RUN: opt %loadPolly %defaultOpts  -polly-analyze-ir  -analyze < %s | FileCheck %s
; RUN: opt %loadPolly %defaultOpts -polly-analyze-ir  -analyze < %s | FileCheck %s
; XFAIL: *
;void f(long a[][128], long N, long M) {
;  long i, j;
;  for (j = 0; j < (4*N + 7*M +3); ++j)
;    for (i = (7*j + 6*M -9); i < (3*j + 5*N + 2) ; ++i)
;        a[i][j] = 0;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f([128 x i64]* nocapture %a, i64 %N, i64 %M) nounwind {
entry:
  %0 = shl i64 %N, 2                              ; <i64> [#uses=2]
  %1 = mul i64 %M, 7                              ; <i64> [#uses=2]
  %2 = or i64 %0, 3                               ; <i64> [#uses=1]
  %3 = add nsw i64 %2, %1                         ; <i64> [#uses=1]
  %4 = icmp sgt i64 %3, 0                         ; <i1> [#uses=1]
  br i1 %4, label %bb.nph8, label %return

bb.nph8:                                          ; preds = %entry
  %tmp14 = mul i64 %M, 6                          ; <i64> [#uses=1]
  %tmp15 = add i64 %tmp14, -9                     ; <i64> [#uses=2]
  %tmp20 = add i64 %1, %0                         ; <i64> [#uses=1]
  %tmp21 = add i64 %tmp20, 3                      ; <i64> [#uses=1]
  %tmp25 = mul i64 %M, -6                         ; <i64> [#uses=1]
  %tmp26 = mul i64 %N, 5                          ; <i64> [#uses=2]
  %tmp27 = add i64 %tmp25, %tmp26                 ; <i64> [#uses=1]
  %tmp28 = add i64 %tmp27, 11                     ; <i64> [#uses=1]
  %tmp35 = add i64 %tmp26, 2                      ; <i64> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb3, %bb.nph8
  %j.07 = phi i64 [ 0, %bb.nph8 ], [ %6, %bb3 ]   ; <i64> [#uses=5]
  %tmp17 = mul i64 %j.07, 897                     ; <i64> [#uses=1]
  %tmp24 = mul i64 %j.07, -4                      ; <i64> [#uses=1]
  %tmp13 = add i64 %tmp24, %tmp28                 ; <i64> [#uses=1]
  %tmp30 = mul i64 %j.07, 7                       ; <i64> [#uses=1]
  %tmp33 = add i64 %tmp30, %tmp15                 ; <i64> [#uses=1]
  %tmp34 = mul i64 %j.07, 3                       ; <i64> [#uses=1]
  %tmp36 = add i64 %tmp34, %tmp35                 ; <i64> [#uses=1]
  %5 = icmp sgt i64 %tmp36, %tmp33                ; <i1> [#uses=1]
  br i1 %5, label %bb1, label %bb3

bb1:                                              ; preds = %bb1, %bb
  %indvar = phi i64 [ 0, %bb ], [ %indvar.next, %bb1 ] ; <i64> [#uses=2]
  %tmp16 = add i64 %indvar, %tmp15                ; <i64> [#uses=1]
  %scevgep = getelementptr [128 x i64]* %a, i64 %tmp16, i64 %tmp17 ; <i64*> [#uses=1]
  store i64 0, i64* %scevgep, align 8
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next, %tmp13    ; <i1> [#uses=1]
  br i1 %exitcond, label %bb3, label %bb1

bb3:                                              ; preds = %bb1, %bb
  %6 = add nsw i64 %j.07, 1                       ; <i64> [#uses=2]
  %exitcond22 = icmp eq i64 %6, %tmp21            ; <i1> [#uses=1]
  br i1 %exitcond22, label %return, label %bb

return:                                           ; preds = %bb3, %entry
  ret void
}

; CHECK: Scop: entry => <Function Return>        Parameters: (%M, %N, ), Max Loop Depth: 2
