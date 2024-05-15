; RUN: opt %loadPolly -polly-opt-isl -S < %s
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"

define void @sdbout_label() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %0 = phi i32 [ 0, %entry ], [ %1, %for.cond ]
  %1 = add nsw i32 %0, 1
  %exitcond72 = icmp eq i32 %1, 7
  br i1 %exitcond72, label %sw.epilog66, label %for.cond

sw.epilog66:                                      ; preds = %for.cond
  ret void
}
