; RUN: opt %loadPolly -polly-codegen < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @main() nounwind {
.split:
  br label %0

.loopexit:                                        ; preds = %.lr.ph, %0
  %indvar.next16 = add i64 %indvar15, 1
  %exitcond53 = icmp eq i64 %indvar.next16, 2048
  br i1 %exitcond53, label %1, label %0

; <label>:0                                       ; preds = %.loopexit, %.split
  %indvar15 = phi i64 [ 0, %.split ], [ %indvar.next16, %.loopexit ]
  %tmp59 = sub i64 2046, %indvar15
  %tmp38 = and i64 %tmp59, 4294967295
  %tmp39 = add i64 %tmp38, 1
  br i1 false, label %.lr.ph, label %.loopexit

.lr.ph:                                           ; preds = %.lr.ph, %0
  %indvar33 = phi i64 [ %indvar.next34, %.lr.ph ], [ 0, %0 ]
  %indvar.next34 = add i64 %indvar33, 1
  %exitcond40 = icmp eq i64 %indvar.next34, %tmp39
  br i1 %exitcond40, label %.loopexit, label %.lr.ph

; <label>:1                                       ; preds = %.loopexit
  ret void
}
