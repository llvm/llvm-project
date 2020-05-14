; RUN: opt %loadPolly -polly-optree-normalize-phi=true -polly-optree -analyze < %s | FileCheck %s -match-full-lines
;
; Rematerialize a load.
;
; for (int j = 0; j < n; j += 1) {
; bodyA:
;   double val = B[j];
;
; bodyB:
;   double phi = val;
;
; bodyC:
;   A[j] = phi;
; }
;
define void @func(i32 %n, double* noalias nonnull %A, double* noalias nonnull %B) {
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
      %phi = phi double [%val, %bodyA]
      br label %bodyC

    bodyC:
      %A_idx = getelementptr inbounds double, double* %A, i32 %j
      store double %phi, double* %A_idx
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
; CHECK:     Reloads: 2
; CHECK: }

; CHECK-NEXT: After statements {
; CHECK-NEXT:     Stmt_bodyA
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_B[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0] -> MemRef_phi__phi[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = load double, double* %B_idx, align 8
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_bodyB
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: [n] -> { Stmt_bodyB[i0] -> MemRef_B[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_phi[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %phi = phi double [ %val, %bodyA ]
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_bodyC
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyC[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyC[i0] -> MemRef_phi[] };
; CHECK-NEXT:            new: [n] -> { Stmt_bodyC[i0] -> MemRef_B[i0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   store double %phi, double* %A_idx
; CHECK-NEXT:             }
; CHECK-NEXT: }
