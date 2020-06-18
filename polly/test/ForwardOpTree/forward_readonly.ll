; RUN: opt %loadPolly -polly-analyze-read-only-scalars=true  -polly-optree -analyze < %s | FileCheck %s -match-full-lines -check-prefixes=STATS,MODEL
; RUN: opt %loadPolly -polly-analyze-read-only-scalars=false -polly-optree -analyze < %s | FileCheck %s -match-full-lines -check-prefixes=STATS,NOMODEL
;
; Move %val to %bodyB, so %bodyA can be removed (by -polly-simplify)
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = arg + 21.0;
;
; bodyB:
;   A[0] = val;
; }
;
define void @func(i32 %n, double* noalias nonnull %A, double %arg) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %val = fadd double %arg, 21.0
      br label %bodyB

    bodyB:
      store double %val, double* %A
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; STATS: Statistics {
; STATS:     Instructions copied: 1
; STATS:     Read-only accesses copied: 1
; STATS:     Operand trees forwarded: 1
; STATS:     Statements with forwarded operand trees: 1
; STATS: }

; MODEL:      After statements {
; MODEL-NEXT:     Stmt_bodyA
; MODEL-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; MODEL-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_arg[] };
; MODEL-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; MODEL-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_val[] };
; MODEL-NEXT:             Instructions {
; MODEL-NEXT:                   %val = fadd double %arg, 2.100000e+01
; MODEL-NEXT:                 }
; MODEL-NEXT:     Stmt_bodyB
; MODEL-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; MODEL-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_A[0] };
; MODEL-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; MODEL-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_arg[] };
; MODEL-NEXT:             Instructions {
; MODEL-NEXT:                   %val = fadd double %arg, 2.100000e+01
; MODEL-NEXT:                   store double %val, double* %A, align 8
; MODEL-NEXT:                 }
; MODEL-NEXT: }

; NOMODEL:      After statements {
; NOMODEL-NEXT:     Stmt_bodyA
; NOMODEL-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; NOMODEL-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_val[] };
; NOMODEL-NEXT:             Instructions {
; NOMODEL-NEXT:                   %val = fadd double %arg, 2.100000e+01
; NOMODEL-NEXT:                 }
; NOMODEL-NEXT:     Stmt_bodyB
; NOMODEL-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; NOMODEL-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_A[0] };
; NOMODEL-NEXT:             Instructions {
; NOMODEL-NEXT:                   %val = fadd double %arg, 2.100000e+01
; NOMODEL-NEXT:                   store double %val, double* %A, align 8
; NOMODEL-NEXT:                 }
; NOMODEL-NEXT: }
