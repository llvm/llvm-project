; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-flatten-schedule -polly-delicm -analyze < %s | FileCheck %s
;
; Use %phi instead of the normal %add; that is, the last last iteration will
; be ignored such the %phi cannot be written to A[3] in %body.
;
;    void func(double *A) {
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        double phi;
;        double incoming = A[j];
;        for (int i = 0; i < 4; i += 1) { /* reduction */
;          phi = incoming;
;          add = phi + 4.2;
;          incoming =  add;
;        }
;        A[j] = phi;
;      }
;    }
;
define void @func(double* noalias nonnull %A) {
entry:
  br label %outer.preheader

outer.preheader:
  br label %outer.for

outer.for:
  %j = phi i32 [0, %outer.preheader], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, 2
  br i1 %j.cmp, label %reduction.preheader, label %outer.exit


    reduction.preheader:
      %A_idx = getelementptr inbounds double, double* %A, i32 %j
      %init = load double, double* %A_idx
      br label %reduction.for

    reduction.for:
      %i = phi i32 [0, %reduction.preheader], [%i.inc, %reduction.inc]
      %phi = phi double [%init, %reduction.preheader], [%add, %reduction.inc]
      br label %body



        body:
          %add = fadd double %phi, 4.2
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      %i.cmp = icmp slt i32 %i.inc, 4
      br i1 %i.cmp, label %reduction.for, label %reduction.exit

    reduction.exit:
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


; CHECK: Statistics {
; CHECK:     Compatible overwrites: 1
; CHECK:     Overwrites mapped to:  1
; CHECK:     Value scalars mapped:  1
; CHECK:     PHI scalars mapped:    1
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_reduction_preheader
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_reduction_preheader[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_preheader[i0] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_preheader[i0] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_reduction_for
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_for[i0, i1] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_for[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_for[i0, i1] -> MemRef_phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_for[i0, i1] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_body
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body[i0, i1] -> MemRef_add[] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body[i0, i1] -> MemRef_phi[] };
; CHECK-NEXT:            new: { Stmt_body[i0, i1] -> MemRef_A[i0] : 3i1 <= 22 - 14i0; Stmt_body[1, 3] -> MemRef_A[1] };
; CHECK-NEXT:     Stmt_reduction_inc
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_add[] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_phi__phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_inc[i0, i1] -> MemRef_A[i0] : i1 <= 2 };
; CHECK-NEXT:     Stmt_reduction_exit
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_reduction_exit[i0] -> MemRef_A[i0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_reduction_exit[i0] -> MemRef_phi[] };
; CHECK-NEXT:            new: { Stmt_reduction_exit[i0] -> MemRef_A[i0] };
; CHECK-NEXT: }
