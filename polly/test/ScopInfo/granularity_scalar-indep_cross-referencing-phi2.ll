; RUN: opt %loadPolly -polly-stmt-granularity=scalar-indep -polly-print-instructions -polly-print-scops -disable-output < %s | FileCheck %s -match-full-lines
;
; Two PHIs, cross-referencing each other. The PHI READs must be carried-out
; before the PHI WRITEs to ensure that the value when entering the block is
; read.
; This means that either both PHIs have to be in the same statement, or the
; PHI WRITEs located in a statement after the PHIs.
;
; for (int j = 0; j < n; j += 1) {
;    double valA = 42.0;
;    double valB = 21.0;
;
; body:
;   double tmp = valA;
;   valA = valB;
;   valB = tmp;
;   A[0] = valA;
; }
;
define void @func(i32 %n, ptr noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %for]
  %valA = phi double [42.0, %entry], [%valB, %for]
  %valB = phi double [21.0, %entry], [%add, %for]
  store double %valB, ptr %A
  %add = fadd double %valA, 0.1
  %j.cmp = icmp slt i32 %j, %n
  %j.inc = add nuw nsw i32 %j, 1
  br i1 %j.cmp, label %for, label %exit

exit:
  br label %return

return:
  ret void
}


; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt_for[i0] : 0 <= i0 <= n; Stmt_for[0] : n < 0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt_for[i0] -> [i0] : i0 <= n; Stmt_for[0] -> [0] : n < 0 };
; CHECK-NEXT:         MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n] -> { Stmt_for[i0] -> MemRef_valA__phi[] };
; CHECK-NEXT:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n] -> { Stmt_for[i0] -> MemRef_valA__phi[] };
; CHECK-NEXT:         MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n] -> { Stmt_for[i0] -> MemRef_valB__phi[] };
; CHECK-NEXT:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [n] -> { Stmt_for[i0] -> MemRef_valB__phi[] };
; CHECK-NEXT:         MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_for[i0] -> MemRef_A[0] };
; CHECK-NEXT:         Instructions {
; CHECK-NEXT:               %valA = phi double [ 4.200000e+01, %entry ], [ %valB, %for ]
; CHECK-NEXT:               %valB = phi double [ 2.100000e+01, %entry ], [ %add, %for ]
; CHECK-NEXT:               store double %valB, ptr %A, align 8
; CHECK-NEXT:               %add = fadd double %valA, 1.000000e-01
; CHECK-NEXT:         }
; CHECK-NEXT: }
