; RUN: opt %loadNPMPolly -passes=polly-codegen -disable-output < %s
target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @main() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvar1 = phi i64 [ %indvar.next2, %for.inc ], [ 0, %entry ] ; <i64> [#uses=2]
  %exitcond = icmp ne i64 %indvar1, 1024          ; <i1> [#uses=1]
  br label %a

a:                                                ; preds = %for.cond
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %a
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvar.next2 = add i64 %indvar1, 1             ; <i64> [#uses=1]
  br label %for.cond

for.end:                                          ; preds = %a
  br label %for.cond5

for.cond5:                                        ; preds = %for.inc07, %for.end
  ret void
}
