; RUN: opt %loadPolly -polly-optree -analyze < %s | FileCheck %s -match-full-lines
;
; Forward a the LoadInst %val into %bodyB. %val is executed multiple times,
; we must get the last loaded values.
;
; for (int j = 0; j < n; j += 1) {
;   double val;
;   for (int i = 0; i < n; i += 1) {
; bodyA:
;     val = B[j];
;   }
;
; bodyB:
;   A[j] = val;
; }
;
define void @func(i32 %n, double* noalias nonnull %A, double* noalias nonnull %B) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp sle i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %i = phi i32 [0, %for], [%i.inc, %bodyA]
      %B_idx = getelementptr inbounds double, double* %B, i32 %i
      %val = load double, double* %B_idx
      %i.inc = add nuw nsw i32 %i, 1
      %i.cmp = icmp slt i32 %i, %n
      br i1 %i.cmp, label %bodyA, label %bodyB

    bodyB:
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
; CHECK:     Known loads forwarded: 1
; CHECK:     Operand trees forwarded: 1
; CHECK:     Statements with forwarded operand trees: 1
; CHECK: }

; CHECK:      After statements {
; CHECK-NEXT:     Stmt_bodyA
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0, i1] -> MemRef_B[i1] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0, i1] -> MemRef_val[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = load double, double* %B_idx
; CHECK-NEXT:                   %i.cmp = icmp slt i32 %i, %n
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_bodyB
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 ;
; CHECK-NEXT:            new: [n] -> { Stmt_bodyB[i0] -> MemRef_B[n] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = load double, double* %B_idx
; CHECK-NEXT:                   store double %val, double* %A_idx
; CHECK-NEXT:             }
; CHECK-NEXT: }
