; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-simplify -analyze < %s | FileCheck -match-full-lines %s
;
; Combine four partial stores into two.
; The stores write to the same array, but never the same element.
;
; for (int j = 0; j < n; j += 1) {
;   A[0] = 21.0;
;   A[1] = 42.0;
;   A[0] = 21.0;
;   A[1] = 42.0;
; }
;
define void @coalesce_disjointelements(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %A_0 = getelementptr inbounds double, double* %A, i32 0
      %A_1 = getelementptr inbounds double, double* %A, i32 1
      store double 21.0, double* %A_0
      store double 42.0, double* %A_1
      store double 21.0, double* %A_0
      store double 42.0, double* %A_1
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
; CHECK:     Partial writes coalesced: 2
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_body
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_body[i0] -> MemRef_A[0] };
; CHECK-NEXT:            new: [n] -> { Stmt_body[i0] -> MemRef_A[0] : n <= 2147483647 };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_body[i0] -> MemRef_A[1] };
; CHECK-NEXT:            new: [n] -> { Stmt_body[i0] -> MemRef_A[1] : n <= 2147483647 };
; CHECK-NEXT: }
