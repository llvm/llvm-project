; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-print-simplify -disable-output < %s | FileCheck -match-full-lines %s
;
; Do not combine stores if there is a read between them.
; Note: The read between is unused, so will be removed by markAndSweep.
; However, searches for coalesces takes place before.
;
; for (int j = 0; j < n; j += 1) {
;   A[0] = 42.0;
;   tmp = A[0];
;   A[0] = 42.0;
; }
;
define void @nocoalesce_readbetween(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      store double 42.0, ptr %A
      %tmp = load double, ptr %A
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
; CHECK:     Partial writes coalesced: 0
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_body
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_body[i0] -> MemRef_A[0] };
; CHECK-NEXT:            new: [n] -> { Stmt_body[i0] -> MemRef_A[0] : i0 >= 17 };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_body[i0] -> MemRef_A[0] };
; CHECK-NEXT:            new: [n] -> { Stmt_body[i0] -> MemRef_A[0] : i0 <= 16 };
; CHECK-NEXT: }
