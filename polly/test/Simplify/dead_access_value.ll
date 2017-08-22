; RUN: opt %loadPolly -polly-simplify -analyze < %s | FileCheck %s -match-full-lines
;
; Remove a dead value write/read pair
; (accesses that are effectively not used)
;
; for (int j = 0; j < n; j += 1) {
; body:
;   double val = 12.5 + 12.5;
; 
; body_succ:
;   double unused = val + 21.0;
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
      %val = fadd double 12.5, 12.5
      br label %body_succ
      
    body_succ:
      %unused = fadd double %val, 21.0
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
; CHECK:     Dead instructions removed: 2
; CHECK:     Stmts removed: 1
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_body_succ
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_body_succ[i0] -> MemRef_A[0] };
; CHECK-NEXT: }
