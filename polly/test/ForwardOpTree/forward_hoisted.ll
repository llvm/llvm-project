; RUN: opt %loadNPMPolly -polly-invariant-load-hoisting=true '-passes=print<polly-optree>' -disable-output < %s | FileCheck %s -match-full-lines
;
; Move %val to %bodyB, so %bodyA can be removed (by -polly-simplify).
; This involves making the load-hoisted %val1 to be made available in %bodyB.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = B[0] + 21.0;
;
; bodyB:
;   A[0] = val;
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
      %val1 = load double, ptr %B
      %val2 = fadd double %val1, 21.0
      br label %bodyB

    bodyB:
      store double %val2, ptr %A
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
; CHECK:     Instructions copied: 1
; CHECK:     Operand trees forwarded: 1
; CHECK:     Statements with forwarded operand trees: 1
; CHECK: }

; CHECK:      After statements {
; CHECK-NEXT:     Stmt_bodyA
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_val2[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val1 = load double, ptr %B, align 8
; CHECK-NEXT:                   %val2 = fadd double %val1, 2.100000e+01
; CHECK-NEXT:                 }
; CHECK-NEXT:     Stmt_bodyB
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_A[0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val2 = fadd double %val1, 2.100000e+01
; CHECK-NEXT:                   store double %val2, ptr %A, align 8
; CHECK-NEXT:                 }
; CHECK-NEXT: }
