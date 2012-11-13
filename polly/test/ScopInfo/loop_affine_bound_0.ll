; RUN: opt %loadPolly %defaultOpts  -polly-analyze-ir  -analyze < %s | FileCheck %s
; RUN: opt %loadPolly %defaultOpts -polly-analyze-ir  -analyze < %s | FileCheck %s
; XFAIL: *

;void f(long a[][128], long N, long M) {
;  long i, j;
;  for (j = 0; j < (4*N + 7*M +3); ++j)
;    for (i = 0; i < (5*N + 2); ++i)
;        ...
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

bb1:                                              ; preds = %bb2.preheader, %bb1
  %i.06 = phi i64 [ 0, %bb2.preheader ], [ %5, %bb1 ] ; <i64> [#uses=2]
  %scevgep = getelementptr [128 x i64]* %a, i64 %i.06, i64 %10 ; <i64*> [#uses=1]
  store i64 0, i64* %scevgep, align 8
  %5 = add nsw i64 %i.06, 1                       ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %5, %8                  ; <i1> [#uses=1]
  br i1 %exitcond, label %bb3, label %bb1

bb3:                                              ; preds = %bb1
  %6 = add i64 %10, 1                             ; <i64> [#uses=2]
  %exitcond14 = icmp eq i64 %6, %tmp13            ; <i1> [#uses=1]
  br i1 %exitcond14, label %return, label %bb2.preheader

bb.nph8:                                          ; preds = %entry
  %7 = mul i64 %N, 5                              ; <i64> [#uses=1]
  %8 = add nsw i64 %7, 2                          ; <i64> [#uses=2]
  %9 = icmp sgt i64 %8, 0                         ; <i1> [#uses=1]
  br i1 %9, label %bb.nph8.split, label %return

bb.nph8.split:                                    ; preds = %bb.nph8
  %tmp12 = add i64 %1, %0                         ; <i64> [#uses=1]
  %tmp13 = add i64 %tmp12, 3                      ; <i64> [#uses=1]
  br label %bb2.preheader

bb2.preheader:                                    ; preds = %bb.nph8.split, %bb3
  %10 = phi i64 [ 0, %bb.nph8.split ], [ %6, %bb3 ] ; <i64> [#uses=2]
  br label %bb1

return:                                           ; preds = %bb.nph8, %bb3, %entry
  ret void
}

; CHECK: entry => <Function Return>        Parameters: (%N, %M, )
