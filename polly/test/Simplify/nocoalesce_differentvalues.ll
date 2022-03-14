; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-print-simplify -disable-output < %s | FileCheck -match-full-lines %s
;
; Do not combine stores that write different values.
;
; for (int j = 0; j < n; j += 1) {
;   A[0] = 21.0;
;   A[0] = 42.0;
; }
;
define void @nocoalesce_differentvalues(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      store double 21.0, double* %A
      store double 42.0, double* %A
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
; CHECK:     Overwrites removed: 0
; CHECK:     Partial writes coalesced: 0
; CHECK: }

; CHECK: SCoP could not be simplified
