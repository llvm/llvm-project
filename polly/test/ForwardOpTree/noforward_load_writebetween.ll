; RUN: opt %loadPolly -polly-print-optree -disable-output < %s | FileCheck %s -match-full-lines
;
; Cannot rematerialize %val from B[0] at bodyC because B[0] has been
; overwritten in bodyB.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = B[j];
;
; bodyB:
;   B[j] = 0.0;
;
; bodyC:
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
      br label %bodyB

    bodyB:
      store double 0.0, ptr %B_idx
      br label %bodyC

    bodyC:
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
