; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-dir=%S -polly-import-jscop-postfix=transformed -polly-simplify -analyze < %s | FileCheck -match-full-lines %s
;
; Remove a store that is overwritten by another store in the same statement.
; Check that this works even if one of the writes is a scalar MemoryKind.
;
; for (int j = 0; j < n; j += 1) {
; body:
;   val = 21.0 + 21.0;
;   A[1] = val;
;
; user:
;   A[0] = val;
; }
;
define void @overwritten_implicit_and_explicit(i32 %n, double* noalias nonnull %A, double* noalias nonnull %C) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %val = fadd double 21.0, 21.0
      store double %val, double* %A
      br label %user

    user:
      store double %val, double* %C
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
; CHECK:     Overwrites removed: 1
; CHECK: }

; CHECK:          Stmt_body
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_body[i0] -> MemRef_val[] };
; CHECK-NEXT:            new: [n] -> { Stmt_body[i0] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_user
