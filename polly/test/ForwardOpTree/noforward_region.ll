; RUN: opt %loadPolly -polly-optree -analyze < %s | FileCheck %s -match-full-lines
;
; Do not move instructions to region statements.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = 21.0 + 21.0;
;
; bodyB_entry:
;   if (undef)
; body_true:
;     A[0] = val;
; }
;
define void @func(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %val = fadd double 21.0, 21.0
      br label %bodyB

    bodyB:
      %cond = fcmp oeq double 21.0, 21.0
      br i1 %cond, label %bodyB_true, label %bodyB_exit

    bodyB_true:
      store double %val, double* %A
      br label %bodyB_exit

    bodyB_exit:
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK: ForwardOpTree executed, but did not modify anything
