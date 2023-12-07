; RUN: opt %loadPolly -polly-print-optree -disable-output < %s | FileCheck %s -match-full-lines
;
; B[j] is overwritten by at least one statement between the
; definition of %val and its use. Hence, it cannot be forwarded.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = B[j];
;   if (j < 1) {
; bodyA_true:
;     B[j] = 0.0;
;   }
;
; bodyB:
;   A[j] = val;
; }
;
define void @func(i32 %n, ptr noalias nonnull %A, ptr noalias nonnull %B) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %B_idx = getelementptr inbounds double, ptr %B, i32 %j
      %val = load double, ptr %B_idx
      %cond = icmp slt i32 %j, 1
      br i1 %cond, label %bodyA_true, label %bodyB

    bodyA_true:
      store double 0.0, ptr %B_idx
      br label %bodyB

    bodyB:
      %A_idx = getelementptr inbounds double, ptr %A, i32 %j
      store double %val, ptr %A_idx
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
