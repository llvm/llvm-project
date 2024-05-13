; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-flatten-schedule -polly-delicm-compute-known=true -polly-delicm-overapproximate-writes=true -polly-delicm-partial-writes=false -polly-print-delicm -disable-output < %s | FileCheck %s --check-prefix=APPROX
; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-flatten-schedule -polly-delicm-compute-known=true -polly-delicm-overapproximate-writes=false -polly-delicm-partial-writes=false  -polly-print-delicm -disable-output < %s | FileCheck %s --check-prefix=EXACT
; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-flatten-schedule -polly-delicm-compute-known=true -polly-delicm-partial-writes=true -polly-print-delicm -disable-output < %s | FileCheck %s --check-prefix=PARTIAL
;
;    void func(double *A {
;      for (int j = -1; j < 3; j += 1) { /* outer */
;        double phi = 0.0;
;        if (0 < j)
;          for (int i = 0; i < j; i += 1) /* reduction */
;            phi += 4.2;
;        A[j] = phi;
;      }
;    }
;
define void @func(ptr noalias nonnull %A) {
entry:
  br label %outer.preheader

outer.preheader:
  br label %outer.for

outer.for:
  %j = phi i32 [-1, %outer.preheader], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, 3
  br i1 %j.cmp, label %reduction.checkloop, label %outer.exit



    reduction.checkloop:
      %j2.cmp = icmp slt i32 0, %j
      br i1 %j2.cmp, label %reduction.preheader, label %reduction.exit

    reduction.preheader:
      br label %reduction.for

    reduction.for:
      %i = phi i32 [0, %reduction.preheader], [%i.inc, %reduction.inc]
      %phi = phi double [0.0, %reduction.preheader], [%add, %reduction.inc]
      br label %body



        body:
          %add = fadd double %phi, 4.2
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      %i.cmp = icmp slt i32 %i.inc, %j
      br i1 %i.cmp, label %reduction.for, label %reduction.exit

    reduction.exit:
      %val = phi double [%add, %reduction.inc], [0.0, %reduction.checkloop]
      %A_idx = getelementptr inbounds double, ptr %A, i32 %j
      store double %val, ptr %A_idx
      br label %outer.inc



outer.inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  ret void
}


; APPROX:      After accesses {
; APPROX-NEXT:     Stmt_reduction_checkloop
; APPROX-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; APPROX-NEXT:                 { Stmt_reduction_checkloop[i0] -> MemRef_val__phi[] };
; APPROX-NEXT:            new: { Stmt_reduction_checkloop[i0] -> MemRef_A[-1 + i0] };
; APPROX-NEXT:     Stmt_reduction_preheader
; APPROX-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; APPROX-NEXT:                 { Stmt_reduction_preheader[i0] -> MemRef_phi__phi[] };
; APPROX-NEXT:            new: { Stmt_reduction_preheader[i0] -> MemRef_A[-1 + i0] };
; APPROX-NEXT:     Stmt_reduction_for
; APPROX-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; APPROX-NEXT:                 { Stmt_reduction_for[i0, i1] -> MemRef_phi__phi[] };
; APPROX-NEXT:            new: { Stmt_reduction_for[i0, i1] -> MemRef_A[-1 + i0] };
; APPROX-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; APPROX-NEXT:                 { Stmt_reduction_for[i0, i1] -> MemRef_phi[] };
; APPROX-NEXT:            new: { Stmt_reduction_for[i0, i1] -> MemRef_A[-1 + i0] };
; APPROX-NEXT:     Stmt_body
; APPROX-NEXT:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; APPROX-NEXT:                { Stmt_body[i0, i1] -> MemRef_add[] };
; APPROX-NEXT:           new: { Stmt_body[i0, i1] -> MemRef_A[-1 + i0] };
; APPROX-NEXT:            ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; APPROX-NEXT:                { Stmt_body[i0, i1] -> MemRef_phi[] };
; APPROX-NEXT:           new: { Stmt_body[i0, i1] -> MemRef_A[-1 + i0] };
; APPROX-NEXT:     Stmt_reduction_inc
; APPROX-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; APPROX-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_add[] };
; APPROX-NEXT:            new: { Stmt_reduction_inc[i0, i1] -> MemRef_A[-1 + i0] };
; APPROX-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; APPROX-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_phi__phi[] };
; APPROX-NEXT:            new: { Stmt_reduction_inc[i0, i1] -> MemRef_A[2] };
; APPROX-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; APPROX-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_val__phi[] };
; APPROX-NEXT:            new: { Stmt_reduction_inc[i0, i1] -> MemRef_A[-1 + i0] };
; APPROX-NEXT:     Stmt_reduction_exit
; APPROX-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; APPROX-NEXT:                 { Stmt_reduction_exit[i0] -> MemRef_val__phi[] };
; APPROX-NEXT:            new: { Stmt_reduction_exit[i0] -> MemRef_A[-1 + i0] };
; APPROX-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; APPROX-NEXT:                 { Stmt_reduction_exit[i0] -> MemRef_A[-1 + i0] };
; APPROX-NEXT: }


; EXACT: No modification has been made


; PARTIAL:      After accesses {
; PARTIAL-NEXT:     Stmt_reduction_checkloop
; PARTIAL-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; PARTIAL-NEXT:                 { Stmt_reduction_checkloop[i0] -> MemRef_val__phi[] };
; PARTIAL-NEXT:            new: { Stmt_reduction_checkloop[i0] -> MemRef_A[-1 + i0] : i0 <= 1 };
; PARTIAL-NEXT:     Stmt_reduction_preheader
; PARTIAL-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; PARTIAL-NEXT:                 { Stmt_reduction_preheader[i0] -> MemRef_phi__phi[] };
; PARTIAL-NEXT:            new: { Stmt_reduction_preheader[i0] -> MemRef_A[-1 + i0] };
; PARTIAL-NEXT:     Stmt_reduction_for
; PARTIAL-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; PARTIAL-NEXT:                 { Stmt_reduction_for[i0, i1] -> MemRef_phi__phi[] };
; PARTIAL-NEXT:            new: { Stmt_reduction_for[i0, i1] -> MemRef_A[-1 + i0] };
; PARTIAL-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; PARTIAL-NEXT:                 { Stmt_reduction_for[i0, i1] -> MemRef_phi[] };
; PARTIAL-NEXT:            new: { Stmt_reduction_for[i0, i1] -> MemRef_A[-1 + i0] };
; PARTIAL-NEXT:     Stmt_body
; PARTIAL-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; PARTIAL-NEXT:                 { Stmt_body[i0, i1] -> MemRef_add[] };
; PARTIAL-NEXT:            new: { Stmt_body[i0, i1] -> MemRef_A[-1 + i0] };
; PARTIAL-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; PARTIAL-NEXT:                 { Stmt_body[i0, i1] -> MemRef_phi[] };
; PARTIAL-NEXT:            new: { Stmt_body[i0, i1] -> MemRef_A[-1 + i0] };
; PARTIAL-NEXT:     Stmt_reduction_inc
; PARTIAL-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; PARTIAL-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_add[] };
; PARTIAL-NEXT:            new: { Stmt_reduction_inc[i0, i1] -> MemRef_A[-1 + i0] };
; PARTIAL-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; PARTIAL-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_phi__phi[] };
; PARTIAL-NEXT:            new: { Stmt_reduction_inc[3, 0] -> MemRef_A[2] };
; PARTIAL-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; PARTIAL-NEXT:                 { Stmt_reduction_inc[i0, i1] -> MemRef_val__phi[] };
; PARTIAL-NEXT:            new: { Stmt_reduction_inc[i0, -2 + i0] -> MemRef_A[-1 + i0] };
; PARTIAL-NEXT:     Stmt_reduction_exit
; PARTIAL-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; PARTIAL-NEXT:                 { Stmt_reduction_exit[i0] -> MemRef_val__phi[] };
; PARTIAL-NEXT:            new: { Stmt_reduction_exit[i0] -> MemRef_A[-1 + i0] };
; PARTIAL-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; PARTIAL-NEXT:                 { Stmt_reduction_exit[i0] -> MemRef_A[-1 + i0] };
; PARTIAL-NEXT: }
