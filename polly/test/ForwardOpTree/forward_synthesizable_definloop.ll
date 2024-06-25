; RUN: opt %loadNPMPolly '-passes=print<polly-optree>' -disable-output < %s | FileCheck %s -match-full-lines
;
; Copy %val to bodyB, assuming the exit value of %i.
;
; for (int j = 0; j < n; j += 1) {
;   double val;
;   for (int i = 0; i < 128; i += 1) {
; bodyA:
;     val = j;
;   }
;
; bodyB:
;   A[0] = val;
; }
;
define void @func(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %inner.for, label %exit

    inner.for:
      %i = phi i32 [0, %for], [%i.inc, %inner.inc]
      br label %bodyA


        bodyA:
          %val = sitofp i32 %i to double
          br label %inner.inc


    inner.inc:
      %i.inc = add nuw nsw i32 %i, 1
      %i.cmp = icmp slt i32 %i.inc, 128
      br i1 %i.cmp, label %inner.for, label %inner.exit

    inner.exit:
       br label %bodyB


    bodyB:
      store double %val, ptr %A
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
; CHECK:     Operand trees forwarded: 1
; CHECK:     Statements with forwarded operand trees: 1
; CHECK: }

; CHECK:      After statements {
; CHECK-NEXT:     Stmt_bodyA
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_bodyA[i0, i1] -> MemRef_val[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = sitofp i32 %i to double
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_bodyB
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_bodyB[i0] -> MemRef_A[0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = sitofp i32 %i to double
; CHECK-NEXT:                   store double %val, ptr %A, align 8
; CHECK-NEXT:             }
; CHECK-NEXT: }
