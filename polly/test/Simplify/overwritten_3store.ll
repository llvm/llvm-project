; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-simplify -analyze < %s | FileCheck -match-full-lines %s 
; RUN: opt %loadPolly -polly-stmt-granularity=bb "-passes=scop(print<polly-simplify>)" -disable-output -aa-pipeline=basic-aa < %s | FileCheck -match-full-lines %s 
;
; Remove a store that is overwritten by another store in the same statement.
; Check that even multiple stores are removed.
;
; for (int j = 0; j < n; j += 1) {
;   A[0] = 10.5;
;   A[0] = 21.0;
;   A[0] = 42.0;
; }
;
define void @overwritten_3store(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      store double 10.5, double* %A
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
; CHECK:     Overwrites removed: 2
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_body
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_body[i0] -> MemRef_A[0] };
; CHECK-NEXT: }
