; RUN: opt %loadNPMPolly -passes=polly-opt-isl -S < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Check that we handle statements with an empty iteration domain correctly.

define void @f() {
entry:
  %A = alloca double
  br label %for

for:
  %indvar = phi i32 [ %indvar.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i32 %indvar, -1
  br i1 %exitcond, label %for.inc, label %return

for.inc:
  %indvar.next = add i32 %indvar, 1
  store double 1.0, ptr %A
  br label %for

return:
  ret void
}
