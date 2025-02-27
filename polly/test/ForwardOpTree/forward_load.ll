; RUN: opt %loadNPMPolly "-passes=scop(print<polly-optree>)" -disable-output < %s | FileCheck %s -match-full-lines
;
; Rematerialize a load.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = B[j];
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


; CHECK: Statistics {
; CHECK:     Known loads forwarded: 1
; CHECK:     Operand trees forwarded: 1
; CHECK:     Statements with forwarded operand trees: 1
; CHECK: }

; CHECK:      Stmt_bodyB
; CHECK-NEXT:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             ;
; CHECK-NEXT:        new: [n] -> { Stmt_bodyB[i0] -> MemRef_B[i0] };
; CHECK-NEXT:         MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_bodyB[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         Instructions {
; CHECK-NEXT:               %val = load double, ptr %B_idx, align 8
; CHECK-NEXT:               store double %val, ptr %A_idx, align 8
; CHECK-NEXT:         }
