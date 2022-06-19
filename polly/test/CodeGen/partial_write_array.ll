; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-codegen -S < %s | FileCheck %s
;
; Partial write of an array access.
;
; for (int j = 0; j < n; j += 1)
;   A[0] = 42.0
;

define void @partial_write_array(i32 %n, double* noalias nonnull %A) {
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


; CHECK:      polly.stmt.body:
; CHECK-NEXT:   %1 = icmp sge i64 %polly.indvar, 5
; CHECK-NEXT:   %polly.Stmt_body_Write0.cond = icmp ne i1 %1, false
; CHECK-NEXT:   br i1 %polly.Stmt_body_Write0.cond, label %polly.stmt.body.Stmt_body_Write0.partial, label %polly.stmt.body.cont

; CHECK:      polly.stmt.body.Stmt_body_Write0.partial:
; CHECK-NEXT:   %polly.access.A = getelementptr double, double* %A, i64 0
; CHECK-NEXT:   store double 4.200000e+01, double* %polly.access.A, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   br label %polly.stmt.body.cont

; CHECK:      polly.stmt.body.cont:
