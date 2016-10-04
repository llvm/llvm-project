; RUN: opt %loadPolly -polly-process-unprofitable=false -polly-unprofitable-scalar-accs=false -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-process-unprofitable=false -polly-unprofitable-scalar-accs=true  -polly-scops -analyze < %s | FileCheck %s --check-prefix=HEURISTIC

; Check the effect of -polly-unprofitable-scalar-accs

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @func(i32 %n, i32 %m, double* noalias nonnull %A) {
entry:
  br label %outer.for

outer.for:
  %j = phi i32 [0, %entry], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %inner.for, label %outer.exit

  inner.for:
    %i = phi i32 [1, %outer.for], [%i.inc, %inner.inc]
    %b = phi double [0.0, %outer.for], [%a, %inner.inc]
    %i.cmp = icmp slt i32 %i, %m
    br i1 %i.cmp, label %body1, label %inner.exit

    body1:
      %A_idx = getelementptr inbounds double, double* %A, i32 %i
      %a = load double, double* %A_idx
      store double %a, double* %A_idx
      br label %inner.inc

  inner.inc:
    %i.inc = add nuw nsw i32 %i, 1
    br label %inner.for

  inner.exit:
    br label %outer.inc

outer.inc:
  store double %b, double* %A
  %j.inc = add nuw nsw i32 %j, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  ret void
}


; CHECK:      Statements {
; CHECK-NEXT:     Stmt_outer_for
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n, m] -> { Stmt_outer_for[i0] : 0 <= i0 <= n; Stmt_outer_for[0] : n < 0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n, m] -> { Stmt_outer_for[i0] -> [i0, 0, 0, 0] : i0 <= n; Stmt_outer_for[0] -> [0, 0, 0, 0] : n < 0 };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n, m] -> { Stmt_outer_for[i0] -> MemRef_b__phi[] };
; CHECK-NEXT:     Stmt_inner_for
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n, m] -> { Stmt_inner_for[i0, i1] : 0 <= i0 < n and 0 <= i1 < m; Stmt_inner_for[i0, 0] : m <= 0 and 0 <= i0 < n };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n, m] -> { Stmt_inner_for[i0, i1] -> [i0, 1, i1, 0] : i1 < m; Stmt_inner_for[i0, 0] -> [i0, 1, 0, 0] : m <= 0 };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n, m] -> { Stmt_inner_for[i0, i1] -> MemRef_b__phi[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n, m] -> { Stmt_inner_for[i0, i1] -> MemRef_b[] };
; CHECK-NEXT:     Stmt_body1
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n, m] -> { Stmt_body1[i0, i1] : 0 <= i0 < n and 0 <= i1 <= -2 + m };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n, m] -> { Stmt_body1[i0, i1] -> [i0, 1, i1, 1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n, m] -> { Stmt_body1[i0, i1] -> MemRef_a[] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n, m] -> { Stmt_body1[i0, i1] -> MemRef_A[1 + i1] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n, m] -> { Stmt_body1[i0, i1] -> MemRef_A[1 + i1] };
; CHECK-NEXT:     Stmt_inner_inc
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n, m] -> { Stmt_inner_inc[i0, i1] : 0 <= i0 < n and 0 <= i1 <= -2 + m };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n, m] -> { Stmt_inner_inc[i0, i1] -> [i0, 1, i1, 2] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n, m] -> { Stmt_inner_inc[i0, i1] -> MemRef_a[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n, m] -> { Stmt_inner_inc[i0, i1] -> MemRef_b__phi[] };
; CHECK-NEXT:     Stmt_outer_inc
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n, m] -> { Stmt_outer_inc[i0] : 0 <= i0 < n };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n, m] -> { Stmt_outer_inc[i0] -> [i0, 2, 0, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n, m] -> { Stmt_outer_inc[i0] -> MemRef_A[0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n, m] -> { Stmt_outer_inc[i0] -> MemRef_b[] };
; CHECK-NEXT: }

; HEURISTIC-NOT: Statements
