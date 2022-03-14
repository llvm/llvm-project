; RUN: opt %loadPolly -polly-print-optree -disable-output < %s | FileCheck %s -match-full-lines
;
; Move instructions to region statements.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = 21.0 + 21.0;
;
; bodyB:
;   if (cond)
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

; CHECK: Statistics {
; CHECK:     Instructions copied: 1
; CHECK:     Known loads forwarded: 0
; CHECK:     Read-only accesses copied: 0
; CHECK:     Operand trees forwarded: 1
; CHECK:     Statements with forwarded operand trees: 1
; CHECK: }

; CHECK: After statements {
; CHECK-NEXT:     Stmt_bodyA
; CHECK-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_val[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = fadd double 2.100000e+01, 2.100000e+01
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_bodyB__TO__bodyB_exit
; CHECK-NEXT:             MayWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB__TO__bodyB_exit[i0] -> MemRef_A[0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = fadd double 2.100000e+01, 2.100000e+01
; CHECK-NEXT:                   %cond = fcmp oeq double 2.100000e+01, 2.100000e+01
; CHECK-NEXT:             }
; CHECK-NEXT: }
