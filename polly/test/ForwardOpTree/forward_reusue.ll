; RUN: opt %loadPolly -polly-print-optree -disable-output < %s | FileCheck %s -match-full-lines
;
; Move operand tree without duplicating values used multiple times.
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
      %val2 = fadd double %val1, %val1
      %val3 = fadd double %val2, %val2
      %val4 = fadd double %val3, %val3
      %val5 = fadd double %val4, %val4
      br label %bodyB

    bodyB:
      store double %val5, double* %A
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
; CHECK:     Instructions copied: 5
; CHECK:     Operand trees forwarded: 1
; CHECK: }

; CHECK:      After statements {
; CHECK-NEXT:     Stmt_bodyA
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_val5[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val1 = fadd double 1.250000e+01, 1.250000e+01
; CHECK-NEXT:                   %val2 = fadd double %val1, %val1
; CHECK-NEXT:                   %val3 = fadd double %val2, %val2
; CHECK-NEXT:                   %val4 = fadd double %val3, %val3
; CHECK-NEXT:                   %val5 = fadd double %val4, %val4
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_bodyB
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_A[0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val1 = fadd double 1.250000e+01, 1.250000e+01
; CHECK-NEXT:                   %val2 = fadd double %val1, %val1
; CHECK-NEXT:                   %val3 = fadd double %val2, %val2
; CHECK-NEXT:                   %val4 = fadd double %val3, %val3
; CHECK-NEXT:                   %val5 = fadd double %val4, %val4
; CHECK-NEXT:                   store double %val5, double* %A, align 8
; CHECK-NEXT:             }
; CHECK-NEXT: }
