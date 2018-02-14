; RUN: opt %loadPolly -polly-delicm -analyze < %s | FileCheck %s -match-full-lines
;
; Reduction over parametric number of elements and a loopguard if the
; reduction loop is not executed at all. Load hoisted before loop.
;
;    void func(double *A) {
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        double phi = 0.0;
;        for (int i = 0; i < n; i += 1) { /* reduction */
;          phi += 4.2;
;          A[j] = phi;
;        }
;      }
;    }
;
define void @func(i32 %n, double* noalias nonnull %A) {
entry:
  br label %outer.preheader

outer.preheader:
  br label %outer.for

outer.for:
  %j = phi i32 [0, %outer.preheader], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, 2
  br i1 %j.cmp, label %reduction.preheader, label %outer.exit


    reduction.preheader:
      %guard.cmp = icmp sle i32 %n,0
      br i1 %guard.cmp, label %reduction.exit, label %reduction.for

    reduction.for:
      %i = phi i32 [0, %reduction.preheader], [%i.inc, %reduction.inc]
      %phi = phi double [0.0, %reduction.preheader], [%add, %reduction.inc]
      br label %body



        body:
          %add = fadd double %phi, 4.2
          %A_idx = getelementptr inbounds double, double* %A, i32 %j
          store double %add, double* %A_idx
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      %i.cmp = icmp slt i32 %i.inc, %n
      br i1 %i.cmp, label %reduction.for, label %reduction.exit

    reduction.exit:
      br label %outer.inc



outer.inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  ret void
}


; CHECK: Statistics {
; CHECK:     Compatible overwrites: 1
; CHECK:     Overwrites mapped to:  1
; CHECK:     Value scalars mapped:  2
; CHECK:     PHI scalars mapped:    1
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_reduction_preheader
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_reduction_preheader[i0] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: [n] -> { Stmt_reduction_preheader[i0] -> MemRef_A[i0] : n > 0 };
; CHECK-NEXT:     Stmt_reduction_for
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_reduction_for[i0, i1] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: [n] -> { Stmt_reduction_for[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_reduction_for[i0, i1] -> MemRef_phi[] };
; CHECK-NEXT:            new: [n] -> { Stmt_reduction_for[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_body
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_body[i0, i1] -> MemRef_add[] };
; CHECK-NEXT:            new: [n] -> { Stmt_body[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_body[i0, i1] -> MemRef_phi[] };
; CHECK-NEXT:            new: [n] -> { Stmt_body[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_body[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_reduction_inc
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_reduction_inc[i0, i1] -> MemRef_add[] };
; CHECK-NEXT:            new: [n] -> { Stmt_reduction_inc[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_reduction_inc[i0, i1] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: [n] -> { Stmt_reduction_inc[i0, i1] -> MemRef_A[i0] : i1 <= -2 + n };
; CHECK-NEXT: }
