; RUN: opt %loadNPMPolly "-passes=scop(print<polly-simplify>)" -disable-output -aa-pipeline=basic-aa < %s | FileCheck -match-full-lines %s
;
; Do not remove overwrites when the value is read before.
;
; for (int j = 0; j < n; j += 1) {
;body:
;   A[0] = 21.0;
;   val = A[0];
;   A[0] = 42.0;
;
;user:
;   B[0] = val;
; }
;
define void @overwritten(i32 %n, ptr noalias nonnull %A, ptr noalias nonnull %B) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      store double 21.0, ptr %A
      %val = load double, ptr %A
      store double 42.0, ptr %A
      br label %user

    user:
      store double %val, ptr %B
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK: SCoP could not be simplified
