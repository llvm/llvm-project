; RUN: opt %loadPolly -polly-import-jscop-dir=%S -polly-import-jscop-postfix=transformed -polly-import-jscop -analyze          < %s | FileCheck %s
; RUN: opt %loadPolly -polly-import-jscop-dir=%S -polly-import-jscop-postfix=transformed -polly-import-jscop -polly-codegen -S < %s | FileCheck %s --check-prefix=CODEGEN

define void @map_scalar_access(double* noalias nonnull %A) {
entry:
  br label %outer.for

outer.for:
  %j = phi i32 [0, %entry], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, 1
  br i1 %j.cmp, label %reduction.for, label %outer.exit


    reduction.for:
      %i = phi i32 [0, %outer.for], [%i.inc, %reduction.inc]
      %phi = phi double [0.0, %outer.for], [%add, %reduction.inc]
      %i.cmp = icmp slt i32 %i, 4
      br i1 %i.cmp, label %body, label %reduction.exit

    body:
      %add = fadd double %phi, 4.2
      br label %reduction.inc

    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      br label %reduction.for

    reduction.exit:
      %A_idx = getelementptr inbounds double, double* %A, i32 %j
      store double %phi, double* %A_idx
      br label %outer.inc


outer.inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  ret void
}



; CHECK:          Arrays {
; CHECK-NEXT:         double MemRef_phi__phi; // Element size 8
; CHECK-NEXT:         double MemRef_phi; // Element size 8
; CHECK-NEXT:         double MemRef_add; // Element size 8
; CHECK-NEXT:         double MemRef_A[*]; // Element size 8
; CHECK-NEXT:     }
; CHECK:          Statements {
; CHECK-NEXT:         Stmt_outer_for
; CHECK-NEXT:             Domain :=
; CHECK-NEXT:                 { Stmt_outer_for[i0] : 0 <= i0 <= 1 };
; CHECK-NEXT:             Schedule :=
; CHECK-NEXT:                 { Stmt_outer_for[i0] -> [i0, 0, 0, 0] };
; CHECK-NEXT:             MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_outer_for[i0] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: { Stmt_outer_for[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         Stmt_reduction_for
; CHECK-NEXT:             Domain :=
; CHECK-NEXT:                 { Stmt_reduction_for[0, i1] : 0 <= i1 <= 4 };
; CHECK-NEXT:             Schedule :=
; CHECK-NEXT:                 { Stmt_reduction_for[i0, i1] -> [0, 1, i1, 0] };
; CHECK-NEXT:             ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_for[i0, i1] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_for[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_for[i0, i1] -> MemRef_phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_for[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:         Stmt_body
; CHECK-NEXT:             Domain :=
; CHECK-NEXT:                 { Stmt_body[0, i1] : 0 <= i1 <= 3 };
; CHECK-NEXT:             Schedule :=
; CHECK-NEXT:                 { Stmt_body[i0, i1] -> [0, 1, i1, 1] };
; CHECK-NEXT:             MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body[i0, i1] -> MemRef_add[] };
; CHECK-NEXT:            new: { Stmt_body[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:             ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body[i0, i1] -> MemRef_phi[] };
; CHECK-NEXT:            new: { Stmt_body[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:         Stmt_reduction_inc
; CHECK-NEXT:             Domain :=
; CHECK-NEXT:                 { Stmt_reduction_inc[0, i1] : 0 <= i1 <= 3 };
; CHECK-NEXT:             Schedule :=
; CHECK-NEXT:                 { Stmt_reduction_inc[i0, i1] -> [0, 1, i1, 2] };
; CHECK-NEXT:             ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_add[] };
; CHECK-NEXT:            new: { Stmt_reduction_inc[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_inc[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:         Stmt_reduction_exit
; CHECK-NEXT:             Domain :=
; CHECK-NEXT:                 { Stmt_reduction_exit[0] };
; CHECK-NEXT:             Schedule :=
; CHECK-NEXT:                 { Stmt_reduction_exit[i0] -> [0, 2, 0, 0] };
; CHECK-NEXT:             MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_reduction_exit[i0] -> MemRef_A[0] };
; CHECK-NEXT:            new: { Stmt_reduction_exit[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_exit[i0] -> MemRef_phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_exit[i0] -> MemRef_A[i0] };
; CHECK-NEXT:     }
; CHECK:      New access function '{ Stmt_outer_for[i0] -> MemRef_A[i0] }' detected in JSCOP file
; CHECK-NEXT: New access function '{ Stmt_reduction_for[i0, i1] -> MemRef_A[i0] }' detected in JSCOP file
; CHECK-NEXT: New access function '{ Stmt_reduction_for[i0, i1] -> MemRef_A[i0] }' detected in JSCOP file
; CHECK-NEXT: New access function '{ Stmt_body[i0, i1] -> MemRef_A[i0] }' detected in JSCOP file
; CHECK-NEXT: New access function '{ Stmt_body[i0, i1] -> MemRef_A[i0] }' detected in JSCOP file
; CHECK-NEXT: New access function '{ Stmt_reduction_inc[i0, i1] -> MemRef_A[i0] }' detected in JSCOP file
; CHECK-NEXT: New access function '{ Stmt_reduction_inc[i0, i1] -> MemRef_A[i0] }' detected in JSCOP file
; CHECK-NEXT: New access function '{ Stmt_reduction_exit[i0] -> MemRef_A[i0] }' detected in JSCOP file
; CHECK-NEXT: New access function '{ Stmt_reduction_exit[i0] -> MemRef_A[i0] }' detected in JSCOP file

; CODEGEN:      polly.stmt.outer.for:
; CODEGEN-NEXT:   %polly.access.A[[R0:[0-9]*]] = getelementptr double, double* %A, i64 %polly.indvar
; CODEGEN-NEXT:   store double 0.000000e+00, double* %polly.access.A[[R0]]
; CODEGEN-NEXT:   br label %polly.cond

; CODEGEN:      polly.stmt.reduction.exit:
; CODEGEN-NEXT:   %polly.access.A[[R1:[0-9]*]] = getelementptr double, double* %A, i64 0
; CODEGEN-NEXT:   %polly.access.A[[R1]].reload = load double, double* %polly.access.A[[R1]]
; CODEGEN-NEXT:   %polly.access.A[[R2:[0-9]*]] = getelementptr double, double* %A, i64 0
; CODEGEN-NEXT:   store double %polly.access.A[[R1]].reload, double* %polly.access.A[[R2]]
; CODEGEN-NEXT:   br label %polly.merge

; CODEGEN:      polly.stmt.reduction.for:
; CODEGEN-NEXT:   %polly.access.A[[R3:[0-9]*]] = getelementptr double, double* %A, i64 0
; CODEGEN-NEXT:   %polly.access.A[[R3]].reload = load double, double* %polly.access.A[[R3]]
; CODEGEN-NEXT:   %polly.access.A[[R4:[0-9]*]] = getelementptr double, double* %A, i64 0
; CODEGEN-NEXT:   store double %polly.access.A[[R3]].reload, double* %polly.access.A[[R4]]
; CODEGEN-NEXT:   br label %polly.cond9

; CODEGEN:      polly.stmt.body:
; CODEGEN-NEXT:   %polly.access.A[[R5:[0-9]*]] = getelementptr double, double* %A, i64 0
; CODEGEN-NEXT:   %polly.access.A[[R5]].reload = load double, double* %polly.access.A[[R5]]
; CODEGEN-NEXT:   %p_add = fadd double %polly.access.A13.reload, 4.200000e+00
; CODEGEN-NEXT:   %polly.access.A[[R6:[0-9]*]] = getelementptr double, double* %A, i64 0
; CODEGEN-NEXT:   store double %p_add, double* %polly.access.A[[R6]]
; CODEGEN-NEXT:   br label %polly.stmt.reduction.inc

; CODEGEN:      polly.stmt.reduction.inc:
; CODEGEN-NEXT:   %polly.access.A[[R7:[0-9]*]] = getelementptr double, double* %A, i64 0
; CODEGEN-NEXT:   %polly.access.A[[R7]].reload = load double, double* %polly.access.A[[R7]]
; CODEGEN-NEXT:   %polly.access.A[[R8:[0-9]*]] = getelementptr double, double* %A, i64 0
; CODEGEN-NEXT:   store double %polly.access.A[[R7]].reload, double* %polly.access.A[[R8]]
; CODEGEN-NEXT:   br label %polly.merge10
