; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-simplify -analyze < %s | FileCheck %s -match-full-lines
;
; Map %val to A[j], so the scalar write on Stmt_for_bodyB can be removed.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = 21.0 + 21.0;
;   A[j] = val;
;
; bodyB:
;   B[j] = val;
; }
;

define void @sweep_mapped_value(i32 %n, double* noalias nonnull %A, double* noalias nonnull %B) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %bodyA, label %exit

    bodyA:
      %val = fadd double 21.0, 21.0
      %A_idx = getelementptr inbounds double, double* %A, i32 %j
      store double %val, double* %A_idx
      br label %bodyB

    bodyB:
      %B_idx = getelementptr inbounds double, double* %B, i32 %j
      store double %val, double* %B_idx
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
; CHECK:     Dead accesses removed: 1
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_bodyA
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_bodyB
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_B[i0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_val[] };
; CHECK-NEXT:            new: [n] -> { Stmt_bodyB[i0] -> MemRef_A[i0] };
; CHECK-NEXT: }
