; RUN: opt %loadPolly -polly-stmt-granularity=scalar-indep -polly-print-instructions -polly-scops -analyze < %s | FileCheck %s -match-full-lines
;
; This case should be split into two statements because {X[0], Y[0]}
; and {A[0], B[0]} do not intersect.
;
;
; for (int j = 0; j < n; j += 1) {
; body:
;   double valX = X[0];
;   Y[0] = valX;
;   double valA = A[0];
;   double valB = B[0];
;   A[0] = valA;
;   A[0] = valB;
; }
;
define void @func(i32 %n, double* noalias nonnull %A, double* noalias nonnull %B, double* noalias nonnull %X, double* noalias nonnull %Y) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %valX = load double, double* %X
      store double %valX, double* %Y
      %valA = load double, double* %A
      %valB = load double, double* %B
      store double %valA, double* %A
      store double %valB, double* %A
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK: Statements {
; CHECK-NEXT: 	Stmt_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt_body[i0] : 0 <= i0 < n };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt_body[i0] -> [i0, 0] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_body[i0] -> MemRef_X[0] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_body[i0] -> MemRef_Y[0] };
; CHECK-NEXT:         Instructions {
; CHECK-NEXT:               %valX = load double, double* %X, align 8
; CHECK-NEXT:               store double %valX, double* %Y, align 8
; CHECK-NEXT:         }
; CHECK-NEXT: 	Stmt_body_b
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt_body_b[i0] : 0 <= i0 < n };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt_body_b[i0] -> [i0, 1] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_body_b[i0] -> MemRef_A[0] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_body_b[i0] -> MemRef_B[0] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_body_b[i0] -> MemRef_A[0] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_body_b[i0] -> MemRef_A[0] };
; CHECK-NEXT:         Instructions {
; CHECK-NEXT:               %valA = load double, double* %A, align 8
; CHECK-NEXT:               %valB = load double, double* %B, align 8
; CHECK-NEXT:               store double %valA, double* %A, align 8
; CHECK-NEXT:               store double %valB, double* %A, align 8
; CHECK-NEXT:         }
; CHECK-NEXT: }
