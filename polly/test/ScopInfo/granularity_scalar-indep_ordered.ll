; RUN: opt %loadPolly -polly-stmt-granularity=scalar-indep -polly-print-instructions -polly-print-scops -disable-output < %s | FileCheck %s -match-full-lines
;
; This case cannot be split into two statements because the order of
; loads and store would be violated.
;
; for (int j = 0; j < n; j += 1) {
; body:
;   double valA = A[0];
;   double valB = B[0];
;   A[0] = valA;
;   A[0] = valB;
; }
;
define void @func(i32 %n, ptr noalias nonnull %A, ptr noalias nonnull %B) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %valA = load double, ptr %A
      %valB = load double, ptr %B
      store double %valA, ptr %A
      store double %valB, ptr %A
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK:     Statements {
; CHECK-NEXT:      Stmt_body
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [n] -> { Stmt_body[i0] : 0 <= i0 < n };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [n] -> { Stmt_body[i0] -> [i0] };
; CHECK-NEXT:          ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:              [n] -> { Stmt_body[i0] -> MemRef_A[0] };
; CHECK-NEXT:          ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:              [n] -> { Stmt_body[i0] -> MemRef_B[0] };
; CHECK-NEXT:          MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:              [n] -> { Stmt_body[i0] -> MemRef_A[0] };
; CHECK-NEXT:          MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:              [n] -> { Stmt_body[i0] -> MemRef_A[0] };
; CHECK-NEXT:          Instructions {
; CHECK-NEXT:                %valA = load double, ptr %A, align 8
; CHECK-NEXT:                %valB = load double, ptr %B, align 8
; CHECK-NEXT:                store double %valA, ptr %A, align 8
; CHECK-NEXT:                store double %valB, ptr %A, align 8
; CHECK-NEXT:          }
; CHECK-NEXT: }
