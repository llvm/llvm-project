; RUN: opt %loadNPMPolly '-passes=print<polly-optree>' -disable-output < %s | FileCheck %s -match-full-lines
;
; Not the entire operand tree can be forwarded,
; some scalar dependencies would remain.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = f() + 21.0;
;
; bodyB:
;   A[0] = val;
; }
;
declare double @f(...) #1

define void @func(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %v = call double (...) @f()
      %val = fadd double %v, 21.0
      br label %bodyB

    bodyB:
      store double %val, ptr %A
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}

attributes #1 = { nounwind readnone }


; CHECK: ForwardOpTree executed, but did not modify anything
