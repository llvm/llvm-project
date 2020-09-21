; RUN: opt %loadPolly -polly-simplify -analyze < %s | FileCheck %s -match-full-lines
; RUN: opt %loadPolly "-passes=scop(print<polly-simplify>)" -disable-output -aa-pipeline=basic-aa < %s | FileCheck %s -match-full-lines
;
; Remove redundant store (a store that writes the same value already
; at the destination)
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
      store double %val, double* %A
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
; CHECK:     Redundant writes removed: 1
; CHECK:     Stmts removed: 1
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT: }
