; RUN: opt %loadPolly -basicaa -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-vectorizer=polly -polly-opt-isl -polly-ast -polly-codegen -S < %s | FileCheck %s
;
; Polly's vectorizer does not support partial accesses.
;
; for (int j = 0; j < 4; j += 1) {
;body:
;   val = 21.0 + 21.0;
;   if (j > 1)
;user:
;     A[0] = val;
; }

define void @partial_write_mapped_vector(double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, 4
  br i1 %j.cmp, label %body, label %exit

    body:
      %val = fadd double 21.0, 21.0
      %if.cond = icmp sgt i32 %j, 1
      br i1 %if.cond, label %user, label %inc

    user:
      %elt= getelementptr inbounds double, double* %A, i32 %j
      store double %val, double* %elt
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
; CHECK-NEXT:    %p_val = fadd double 2.100000e+01, 2.100000e+01
; CHECK-NEXT:    %0 = trunc i64 %polly.indvar to i32
; CHECK-NEXT:    %p_if.cond = icmp sgt i32 %0, 1
; CHECK-NEXT:    %1 = icmp sge i64 %polly.indvar, 2
; CHECK-NEXT:    %polly.Stmt_body_Write0.cond = icmp ne i1 %1, false
; CHECK-NEXT:    br i1 %polly.Stmt_body_Write0.cond, label %polly.stmt.body.Stmt_body_Write0.partial, label %polly.stmt.body.cont

; CHECK-LABEL:  polly.stmt.body.Stmt_body_Write0.partial:
; CHECK-NEXT:    %polly.access.A = getelementptr double, double* %A, i64 1
; CHECK-NEXT:    store double %p_val, double* %polly.access.A
; CHECK-NEXT:    br label %polly.stmt.body.cont

; CHECK-LABEL:  polly.stmt.body.cont:
