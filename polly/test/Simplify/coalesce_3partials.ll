; RUN: opt %loadNPMPolly -polly-stmt-granularity=bb '-passes=polly-import-jscop,print<polly-simplify>' -polly-import-jscop-postfix=transformed -disable-output < %s | FileCheck -match-full-lines %s
;
; Combine 3 partial accesses into one.
;
; for (int j = 0; j < n; j += 1) {
;   A[0] = 42.0;
;   A[0] = 42.0;
;   A[0] = 42.0;
; }
;
define void @coalesce_3partials(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      store double 42.0, ptr %A
      store double 42.0, ptr %A
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
; CHECK:     Overwrites removed: 0
; CHECK:     Partial writes coalesced: 2
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_body
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_body[i0] -> MemRef_A[0] };
; CHECK-NEXT:            new: [n] -> { Stmt_body[i0] -> MemRef_A[0] };
; CHECK-NEXT: }
