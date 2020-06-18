; RUN: opt %loadPolly -polly-optree -analyze < %s | FileCheck %s -match-full-lines
;
; Move %v and %val to %bodyB, so %bodyA can be removed (by -polly-simplify)
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val1 = 12.5 + 12.5;
;
; bodyB:
;   double val2 = 21.0 + 21.0;
;
; bodyC:
;   A[0] = val2;
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
      %val1 = fadd double 12.5, 12.5
      br label %bodyB

    bodyB:
      %val2 = fadd double %val1, 21.0
      br label %bodyC

    bodyC:
      store double %val2, double* %A
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
; CHECK:     Instructions copied: 3
; CHECK:     Operand trees forwarded: 2
; CHECK:     Statements with forwarded operand trees: 2
; CHECK: }

; CHECK:      After statements {
; CHECK-NEXT:     Stmt_bodyA
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_val1[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val1 = fadd double 1.250000e+01, 1.250000e+01
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_bodyB
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_val2[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val1 = fadd double 1.250000e+01, 1.250000e+01
; CHECK-NEXT:                   %val2 = fadd double %val1, 2.100000e+01
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_bodyC
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyC[i0] -> MemRef_A[0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val1 = fadd double 1.250000e+01, 1.250000e+01
; CHECK-NEXT:                   %val2 = fadd double %val1, 2.100000e+01
; CHECK-NEXT:                   store double %val2, double* %A, align 8
; CHECK-NEXT:             }
; CHECK-NEXT: }
