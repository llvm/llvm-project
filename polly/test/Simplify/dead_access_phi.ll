; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-simplify -analyze < %s | FileCheck %s -match-full-lines
; RUN: opt %loadPolly -polly-stmt-granularity=bb "-passes=scop(print<polly-simplify>)" -disable-output -aa-pipeline=basic-aa < %s | FileCheck %s -match-full-lines
;
; Remove a dead PHI write/read pair
; (accesses that are effectively not used)
;
; for (int j = 0; j < n; j += 1) {
; body:
;   double phi = 42;
;
; body_succ:
;   A[0] = 42.0;
; }
;
define void @func(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      br label %body_succ

    body_succ:
      %phi = phi double [42.0, %body]
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
; CHECK:     Dead accesses removed: 2
; CHECK:     Dead instructions removed: 1
; CHECK:     Stmts removed: 1
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_body_succ
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_body_succ[i0] -> MemRef_A[0] };
; CHECK-NEXT: }
