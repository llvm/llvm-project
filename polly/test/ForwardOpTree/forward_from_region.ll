; RUN: opt %loadPolly -polly-print-optree -disable-output < %s | FileCheck %s -match-full-lines
;
; Move instructions from region statements.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = 21.0 + 21.0;
;   if (cond)
;
; bodyA_true:
;     A[0] = 42;
;
; bodyB:
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
      %cond = fcmp oeq double 21.0, 21.0
      br i1 %cond, label %bodyA_true, label %bodyB

    bodyA_true:
      store double 42.0, double* %A
      br label %bodyB

    bodyB:
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

; CHECK:      Statistics {
; CHECK:     Instructions copied: 1
; CHECK:     Known loads forwarded: 0
; CHECK:     Read-only accesses copied: 0
; CHECK:     Operand trees forwarded: 1
; CHECK:     Statements with forwarded operand trees: 1
; CHECK: }
; CHECK: After statements {
; CHECK:     Stmt_bodyA__TO__bodyB
; CHECK:             MayWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK:                 [n] -> { Stmt_bodyA__TO__bodyB[i0] -> MemRef_A[0] };
; CHECK:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK:                 [n] -> { Stmt_bodyA__TO__bodyB[i0] -> MemRef_val[] };
; CHECK:             Instructions {
; CHECK:                   %val = fadd double 2.100000e+01, 2.100000e+01
; CHECK:                   %cond = fcmp oeq double 2.100000e+01, 2.100000e+01
; CHECK:             }
; CHECK:     Stmt_bodyB
; CHECK:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK:                 [n] -> { Stmt_bodyB[i0] -> MemRef_A[0] };
; CHECK:             Instructions {
; CHECK:                   %val = fadd double 2.100000e+01, 2.100000e+01
; CHECK:                   store double %val, double* %A, align 8
; CHECK:             }
; CHECK: }

