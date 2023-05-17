; RUN: opt %loadPolly -polly-import-jscop-postfix=transformed -polly-print-import-jscop -polly-print-simplify -disable-output < %s | FileCheck %s -match-full-lines
;
; Remove a redundant store, if its partial domain is a subset of the
; read's domain.
;
define void @redundant_partialwrite(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %val = load double, ptr %A
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


; Check successful import.
; CHECK:    new: [n] -> { Stmt_body[i0] -> MemRef_A[0] : i0 <= 15 };

; CHECK: Statistics {
; CHECK:     Redundant writes removed: 1
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT: }
