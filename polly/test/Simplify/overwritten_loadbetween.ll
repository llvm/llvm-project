; RUN: opt %loadPolly -polly-simplify -analyze < %s | FileCheck -match-full-lines %s 
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
define void @overwritten(i32 %n, double* noalias nonnull %A, double* noalias nonnull %B) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      store double 21.0, double* %A
      %val = load double, double* %A
      store double 42.0, double* %A
      br label %user

    user:
      store double %val, double* %B
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
