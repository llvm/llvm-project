; RUN: opt %loadPolly %defaultOpts -polly-cloog -analyze   -analyze < %s | not FileCheck %s

;void f(long a[][128], long N, long M) {
;  long i, j;
;  for (j = 0; j < M; ++j)
;    for (i = 0; i < rnd(); ++i)
;        ...
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f([128 x i64]* nocapture %a, i64 %N, i64 %M) nounwind {
entry:
  %0 = icmp sgt i64 %M, 0                         ; <i1> [#uses=1]
  br i1 %0, label %bb2.preheader, label %return

bb1:                                              ; preds = %bb2.preheader, %bb1
  %i.06 = phi i64 [ 0, %bb2.preheader ], [ %1, %bb1 ] ; <i64> [#uses=3]
  %scevgep = getelementptr [128 x i64]* %a, i64 %i.06, i64 %5 ; <i64*> [#uses=1]
  %tmp = add i64 %i.06, %N                        ; <i64> [#uses=1]
  store i64 %tmp, i64* %scevgep, align 8
  %1 = add nsw i64 %i.06, 1                       ; <i64> [#uses=2]
  %2 = tail call i64 (...)* @rnd() nounwind       ; <i64> [#uses=1]
  %3 = icmp sgt i64 %2, %1                        ; <i1> [#uses=1]
  br i1 %3, label %bb1, label %bb3

bb3:                                              ; preds = %bb2.preheader, %bb1
  %4 = add i64 %5, 1                              ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %4, %M                  ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb2.preheader

bb2.preheader:                                    ; preds = %bb3, %entry
  %5 = phi i64 [ %4, %bb3 ], [ 0, %entry ]        ; <i64> [#uses=2]
  %6 = tail call i64 (...)* @rnd() nounwind       ; <i64> [#uses=1]
  %7 = icmp sgt i64 %6, 0                         ; <i1> [#uses=1]
  br i1 %7, label %bb1, label %bb3

return:                                           ; preds = %bb3, %entry
  ret void
}

declare i64 @rnd(...)

; CHECK: Scop!
