; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-optree -polly-codegen -analyze < %s | FileCheck %s -match-full-lines
;
; %val1 is used three times: Twice by its own operand tree of %val2 and once
; more by the store in %bodyB.
; Verify that we can handle multiple uses by the same instruction and uses
; in multiple statements as well.
; The result processing may depend on the order in which the values are used,
; hence we check both orderings.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val1 = A[j];
;   double val2 = val1 + val1;
;
; bodyB:
;   B[j] = val1;
;   C[j] = val2;
; }
;
define void @func1(i32 %n, double* noalias nonnull %A, double* noalias nonnull %B, double* noalias nonnull %C) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %A_idx = getelementptr inbounds double, double* %A, i32 %j
      %val1 = load double, double* %A_idx
      %val2 = fadd double %val1, %val1
      br label %bodyB

    bodyB:
      %B_idx = getelementptr inbounds double, double* %B, i32 %j
      store double %val1, double* %B_idx
      %C_idx = getelementptr inbounds double, double* %C, i32 %j
      store double %val2, double* %C_idx
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
; CHECK:     Known loads forwarded: 3
; CHECK:     Operand trees forwarded: 2
; CHECK:     Statements with forwarded operand trees: 1
; CHECK: }

; CHECK:      After statements {
; CHECK-NEXT:     Stmt_bodyA
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_val1[] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_val2[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val1 = load double, double* %A_idx, align 8
; CHECK-NEXT:                   %val2 = fadd double %val1, %val1
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_bodyB
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 ;
; CHECK-NEXT:            new: [n] -> { Stmt_bodyB[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_B[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_C[i0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val1 = load double, double* %A_idx, align 8
; CHECK-NEXT:                   %val1 = load double, double* %A_idx, align 8
; CHECK-NEXT:                   %val2 = fadd double %val1, %val1
; CHECK-NEXT:                   %val1 = load double, double* %A_idx, align 8
; CHECK-NEXT:                   store double %val1, double* %B_idx, align 8
; CHECK-NEXT:                   store double %val2, double* %C_idx, align 8
; CHECK-NEXT:             }
; CHECK-NEXT: }


define void @func2(i32 %n, double* noalias nonnull %A, double* noalias nonnull %B, double* noalias nonnull %C) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %A_idx = getelementptr inbounds double, double* %A, i32 %j
      %val1 = load double, double* %A_idx
      %val2 = fadd double %val1, %val1
      br label %bodyB

    bodyB:
      %B_idx = getelementptr inbounds double, double* %B, i32 %j
      store double %val2, double* %B_idx
      %C_idx = getelementptr inbounds double, double* %C, i32 %j
      store double %val1, double* %C_idx
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
; CHECK:     Known loads forwarded: 3
; CHECK:     Operand trees forwarded: 2
; CHECK:     Statements with forwarded operand trees: 1
; CHECK: }

; CHECK:      After statements {
; CHECK-NEXT:     Stmt_bodyA
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_val2[] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_val1[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val1 = load double, double* %A_idx, align 8
; CHECK-NEXT:                   %val2 = fadd double %val1, %val1
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_bodyB
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 ;
; CHECK-NEXT:            new: [n] -> { Stmt_bodyB[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_B[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_C[i0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val1 = load double, double* %A_idx, align 8
; CHECK-NEXT:                   %val1 = load double, double* %A_idx, align 8
; CHECK-NEXT:                   %val1 = load double, double* %A_idx, align 8
; CHECK-NEXT:                   %val2 = fadd double %val1, %val1
; CHECK-NEXT:                   store double %val2, double* %B_idx, align 8
; CHECK-NEXT:                   store double %val1, double* %C_idx, align 8
; CHECK-NEXT:             }
; CHECK-NEXT: }
