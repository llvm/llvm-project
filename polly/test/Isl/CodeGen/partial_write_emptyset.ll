; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-codegen -S < %s | FileCheck %s
;
; Partial write, where "partial" is the empty set.
; The store is never executed in this case and we do generate it in the
; first place.
;
; for (int j = 0; j < n; j += 1)
;   A[0] = 42.0
;

define void @partial_write_emptyset(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
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


; CHECK-LABEL: polly.stmt.body:
; CHECK-NOT:     store
