; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb '-passes=polly-import-jscop,print<polly-simplify>' -polly-import-jscop-postfix=transformed -disable-output < %s | FileCheck -match-full-lines %s
;
; Do not combine stores that do not write to different elements in the
; same instance.
;
; for (int j = 0; j < n; j += 1) {
;   A[0] = 21.0;
;   A[0] = 42.0;
; }
;
define void @nocoalesce_elementmismatch(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %A_1 = getelementptr inbounds double, ptr %A, i32 1
      store double 42.0, ptr %A
      store double 42.0, ptr %A_1
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
