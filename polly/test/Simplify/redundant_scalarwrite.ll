; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-print-simplify -disable-output < %s | FileCheck %s -match-full-lines
;
; Remove redundant scalar stores.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = A[0];
;
; bodyB:
;   A[0] = val;
; }
;
define void @redundant_scalarwrite(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit


    bodyA:
      %val = load double, ptr %A
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


; CHECK: Statistics {
; CHECK:     Redundant writes removed: 2
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT: }
