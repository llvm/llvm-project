; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s 2>&1 | not FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@edge.8265 = external global [72 x i32], align 32 ; <ptr> [#uses=1]

define void @compact_unitcell_edges() nounwind {
bb.nph19:
  br label %bb4

bb4:                                              ; preds = %bb4, %bb.nph19
  %e.118 = phi i32 [ 0, %bb.nph19 ], [ %tmp23, %bb4 ] ; <i32> [#uses=1]
  %i.017 = phi i32 [ 0, %bb.nph19 ], [ %0, %bb4 ] ; <i32> [#uses=1]
  %tmp23 = add i32 %e.118, 8                      ; <i32> [#uses=2]
  %0 = add nsw i32 %i.017, 1                      ; <i32> [#uses=2]
  %exitcond42 = icmp eq i32 %0, 6                 ; <i1> [#uses=1]
  br i1 %exitcond42, label %bb.nph, label %bb4

bb.nph:                                           ; preds = %bb4
  %tmp = sext i32 %tmp23 to i64                   ; <i64> [#uses=1]
  br label %bb7

bb7:                                              ; preds = %bb7, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb7 ] ; <i64> [#uses=2]
  %tmp21 = add i64 %tmp, %indvar                  ; <i64> [#uses=1]
  %scevgep = getelementptr [72 x i32], ptr @edge.8265, i64 0, i64 %tmp21 ; <ptr> [#uses=1]
  store i32 undef, ptr %scevgep, align 4
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br i1 undef, label %bb10, label %bb7

bb10:                                             ; preds = %bb7
  ret void
}

; CHECK: SCOP:
