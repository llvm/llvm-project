; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-print-simplify -disable-output < %s | FileCheck %s -match-full-lines
; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb "-passes=scop(print<polly-simplify>)" -disable-output -aa-pipeline=basic-aa < %s | FileCheck %s -match-full-lines
;
; Remove a dead load-instruction
; (an load whose result is not used anywhere)
;
; for (int j = 0; j < n; j += 1) {
;   double val = A[0];
;   A[0] = 42.0;
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
      %val = load double, double* %A
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
; CHECK:     Dead accesses removed: 1
; CHECK:     Dead instructions removed: 1
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_body
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_body[i0] -> MemRef_A[0] };
; CHECK-NEXT: }
