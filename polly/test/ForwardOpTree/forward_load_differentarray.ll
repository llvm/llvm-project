; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-optree -analyze < %s | FileCheck %s -match-full-lines
;
; To forward %val, B[j] cannot be reused in bodyC because it is overwritten
; between. Verify that instead the alternative C[j] is used.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = B[j];
;
; bodyB:
;   B[j] = 0;
;   C[j] = val;
;
; bodyC:
;   A[j] = val;
; }
;
define void @func(i32 %n, double* noalias nonnull %A, double* noalias nonnull %B, double* noalias nonnull %C) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %B_idx = getelementptr inbounds double, double* %B, i32 %j
      %val = load double, double* %B_idx
      br label %bodyB

    bodyB:
      store double 0.0, double* %B_idx
      %C_idx = getelementptr inbounds double, double* %C, i32 %j
      store double %val, double* %C_idx
      br label %bodyC

    bodyC:
      %A_idx = getelementptr inbounds double, double* %A, i32 %j
      store double %val, double* %A_idx
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
; CHECK:     Known loads forwarded: 2
; CHECK:     Operand trees forwarded: 2
; CHECK:     Statements with forwarded operand trees: 2
; CHECK: }

; CHECK-NEXT: After statements {
; CHECK-NEXT:     Stmt_bodyA
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_B[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_val[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = load double, double* %B_idx, align 8
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_bodyB
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 ;
; CHECK-NEXT:            new: [n] -> { Stmt_bodyB[i0] -> MemRef_B[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_B[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_C[i0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = load double, double* %B_idx, align 8
; CHECK-NEXT:                   store double 0.000000e+00, double* %B_idx
; CHECK-NEXT:                   store double %val, double* %C_idx
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_bodyC
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 ;
; CHECK-NEXT:            new: [n] -> { Stmt_bodyC[i0] -> MemRef_C[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyC[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = load double, double* %B_idx, align 8
; CHECK-NEXT:                   store double %val, double* %A_idx
; CHECK-NEXT:             }
; CHECK-NEXT: }
