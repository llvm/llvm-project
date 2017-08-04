; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-codegen -S < %s | FileCheck %s
;
; Partial write of a (mapped) scalar in a non-affine subregion.
;
; for (int j = 0; j < n; j += 1) {
;subregion:
;   val = 21.0 + 21.0;
;   if (undef > undef)
;subregion_true: ;
;
;subregion_exit:
;   if (j >= 5)
;user:
;     A[0] = val;
; }

define void @partial_write_mapped_scalar_subregion(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %subregion, label %exit

    subregion:
      %val = fadd double 21.0, 21.0
      %nonaffine.cond = fcmp ogt double undef, undef
      br i1 %nonaffine.cond, label %subregion_true, label %subregion_exit

    subregion_true:
      br label %subregion_exit

    subregion_exit:
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


; CHECK-LABEL: polly.stmt.subregion_exit.exit:
; CHECK-NEXT:    %1 = icmp sge i64 %polly.indvar, 5
; CHECK-NEXT:    %polly.Stmt_subregion__TO__subregion_exit_Write0.cond = icmp ne i1 %1, false
; CHECK-NEXT:    br i1 %polly.Stmt_subregion__TO__subregion_exit_Write0.cond, label %polly.stmt.subregion_exit.exit.Stmt_subregion__TO__subregion_exit_Write0.partial, label %polly.stmt.subregion_exit.exit.cont

; CHECK-LABEL: polly.stmt.subregion_exit.exit.Stmt_subregion__TO__subregion_exit_Write0.partial:
; CHECK-NEXT:    %polly.access.A = getelementptr double, double* %A, i64 1
; CHECK-NEXT:    store double %p_val, double* %polly.access.A
; CHECK-NEXT:    br label %polly.stmt.subregion_exit.exit.cont

; CHECK-LABEL: polly.stmt.subregion_exit.exit.cont:
