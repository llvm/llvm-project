; RUN: opt %loadNPMPolly -passes=polly-codegen < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @CleanNet() nounwind {
entry:
  %firstVia.0.reg2mem = alloca i64
  br label %bb7

bb7:                                              ; preds = %bb7, %entry
  br i1 undef, label %bb7, label %bb8

bb8:                                              ; preds = %bb7
  %indvar5.lcssa.reload = load i64, ptr undef
  %tmp17 = mul i64 %indvar5.lcssa.reload, -1
  %tmp18 = add i64 0, %tmp17
  br label %bb18

bb13:                                             ; preds = %bb18
  %0 = icmp ult i64 %i.1, 0
  br i1 %0, label %bb14, label %bb17

bb14:                                             ; preds = %bb13
  store i64 %i.1, ptr %firstVia.0.reg2mem
  br label %bb17

bb17:                                             ; preds = %bb14, %bb13
  %indvar.next16 = add i64 %indvar15, 1
  br label %bb18

bb18:                                             ; preds = %bb17, %bb8
  %indvar15 = phi i64 [ %indvar.next16, %bb17 ], [ 0, %bb8 ]
  %i.1 = add i64 %tmp18, %indvar15
  br i1 undef, label %bb13, label %bb25

bb25:                                             ; preds = %bb18
  ret void
}
