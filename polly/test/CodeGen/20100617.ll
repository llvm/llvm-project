; RUN: opt %loadPolly -polly-codegen < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @init_array() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond1, %entry
  %indvar1 = phi i64 [ %indvar.next2, %for.cond1 ], [ 0, %entry ] ; <i64> [#uses=1]
  br i1 false, label %for.cond1, label %for.end32

for.cond1:                                        ; preds = %for.cond
  %indvar.next2 = add i64 %indvar1, 1             ; <i64> [#uses=1]
  br label %for.cond

for.end32:                                        ; preds = %for.cond
  ret void
}
