; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-dir=%S -polly-import-jscop-postfix=transformed -polly-simplify -analyze < %s | FileCheck -match-full-lines %s
;
; Remove identical writes
; (two stores in the same statement that write the same value to the same
; destination)
;
; for (int j = 0; j < n; j += 1) {
; body:
;   val = 21.0 + 21.0;
;   A[1] = val;
;   A[1] = val;
;   A[1] = val;
;
; user:
;   A[0] = A[1];
; }
;
define void @overwritten_3phi(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %val = fadd double 21.0, 21.0
      br label %user

    user:
      %phi1 = phi double [%val, %body]
      %phi2 = phi double [%val, %body]
      %phi3 = phi double [%val, %body]
      %add1 = fadd double %phi1, %phi2
      %add2 = fadd double %add1, %phi3
      store double %add2, double* %A
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

; CHECK:      Stmt_body
; CHECK-NEXT:         MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n] -> { Stmt_body[i0] -> MemRef_phi3__phi[] };
; CHECK-NEXT:        new: [n] -> { Stmt_body[i0] -> MemRef_A[1] };
; CHECK-NEXT: Stmt_user
