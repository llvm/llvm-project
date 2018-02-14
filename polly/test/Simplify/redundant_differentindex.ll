; RUN: opt %loadPolly -polly-simplify -analyze < %s | FileCheck %s -match-full-lines
;
; A store that has a different index than the load it is storing is
; not redundant.
;
; for (int j = 0; j < n; j += 1)
;   A[0] = A[0];
;
define void @func(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %val = load double, double* %A
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


; CHECK: SCoP could not be simplified
