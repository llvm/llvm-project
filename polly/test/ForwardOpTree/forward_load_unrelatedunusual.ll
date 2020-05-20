; RUN: opt %loadPolly -polly-optree -analyze < %s | FileCheck %s -match-full-lines
;
; Rematerialize a load.
; The non-analyzable store to C[0] is unrelated and can be ignored.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = B[j];
;   C[0] = 21.0;
;   C[0] = 42.0;
;
; bodyB:
;   A[j] = val;
; }
;
define void @func(i32 %n, double* noalias nonnull %A, double* noalias nonnull %B, double *noalias %C) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %B_idx = getelementptr inbounds double, double* %B, i32 %j
      %val = load double, double* %B_idx
      store double 21.0, double* %C
      store double 41.0, double* %C
      br label %bodyB

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

; CHECK:      Stmt_bodyB
; CHECK-NEXT:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             ;
; CHECK-NEXT:        new: [n] -> { Stmt_bodyB[i0] -> MemRef_B[i0] };
; CHECK-NEXT:         MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_bodyB[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         Instructions {
; CHECK-NEXT:               %val = load double, double* %B_idx
; CHECK-NEXT:               store double %val, double* %A_idx
; CHECK-NEXT:         }
