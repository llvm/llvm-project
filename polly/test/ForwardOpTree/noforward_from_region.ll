; RUN: opt %loadPolly -polly-print-optree -disable-output < %s | FileCheck %s -match-full-lines
;
; Ensure we do not move instructions from region statements in case the
; instruction to move loads from an array which is also written to from
; within the region. This is necessary as complex region statements may prevent
; us from detecting possible memory conflicts.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = A[0];
;   if (cond)
;
; bodyA_true:
;     A[0] = 42;
;
; bodyB:
;     A[0] = val;
; }
;
define void @func(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %val = load double, ptr %A
      %cond = fcmp oeq double 21.0, 21.0
      br i1 %cond, label %bodyA_true, label %bodyB

    bodyA_true:
      store double 42.0, ptr %A
      br label %bodyB

    bodyB:
      store double %val, ptr %A
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
