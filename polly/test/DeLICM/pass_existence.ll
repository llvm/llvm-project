; RUN: opt %loadPolly -polly-delicm -analyze < %s | FileCheck %s
; RUN: opt %loadPolly "-passes=scop(print<polly-delicm>)" -disable-output < %s | FileCheck %s
;
; Simple test for the existence of the DeLICM pass.
;
; // Simplest detected SCoP to run DeLICM on.
; for (int j = 0; j < n; j += 1) {
;   body: A[0] = 0.0;
; }
;
define void @func(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      store double 0.0, double* %A
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; Verify that the DeLICM has a custom printScop() function.
; CHECK: DeLICM result:
