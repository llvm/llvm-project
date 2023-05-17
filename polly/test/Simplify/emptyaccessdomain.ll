; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-print-simplify -disable-output < %s | FileCheck %s -match-full-lines
;
; for (int j = 0; j < n; j += 1) {
;   A[0] = 42.0;
; }
;
define void @emptyaccessdomain(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      store double 42.0, ptr %A
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
; CHECK:     Accesses with empty domains removed: 1
; CHECK:     Stmts removed: 1
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT: }
