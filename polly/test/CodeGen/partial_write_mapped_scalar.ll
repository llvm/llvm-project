; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-codegen -S < %s | FileCheck %s
;
; Partial write of a (mapped) scalar.
;
; for (int j = 0; j < n; j += 1) {
;body:
;   val = 21.0 + 21.0;
;   if (j >= 5)
;user:
;     A[0] = val;
; }

define void @partial_write_mapped_scalar(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %val = fadd double 21.0, 21.0
      %if.cond = icmp sgt i32 %j, 5
      br i1 %if.cond, label %user, label %inc

    user:
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


; CHECK:      polly.stmt.body:
; CHECK-NEXT:   %p_val = fadd double 2.100000e+01, 2.100000e+01
; CHECK-NEXT:   %1 = trunc i64 %polly.indvar to i32
; CHECK-NEXT:   %p_if.cond = icmp sgt i32 %1, 5
; CHECK-NEXT:   %2 = icmp sge i64 %polly.indvar, 5
; CHECK-NEXT:   %polly.Stmt_body_Write0.cond = icmp ne i1 %2, false
; CHECK-NEXT:   br i1 %polly.Stmt_body_Write0.cond, label %polly.stmt.body.Stmt_body_Write0.partial, label %polly.stmt.body.cont

; CHECK:      polly.stmt.body.Stmt_body_Write0.partial:
; CHECK-NEXT:   %polly.access.A = getelementptr double, double* %A, i64 1
; CHECK-NEXT:   store double %p_val, double* %polly.access.A
; CHECK-NEXT:   br label %polly.stmt.body.cont

; CHECK:      polly.stmt.body.cont:
; CHECK-NEXT:   br label %polly.cond
