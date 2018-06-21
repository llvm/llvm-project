; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-delicm -analyze< %s | FileCheck %s
; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-delicm -disable-output -stats < %s 2>&1 | FileCheck %s --check-prefix=STATS
; REQUIRES: asserts
;
;    void func(double *A) {
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        A[j] = 21.0;
;        A[j] = 42.0;
;        double phi = 0.0;
;        for (int i = 0; i < 4; i += 1) /* reduction */
;          phi += 4.2;
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
      store double 21.0, double* %A_idx
      store double 42.0, double* %A_idx
      br label %reduction.for

    reduction.for:
      %i = phi i32 [0, %reduction.preheader], [%i.inc, %reduction.inc]
      %phi = phi double [0.0, %reduction.preheader], [%add, %reduction.inc]
      %i.cmp = icmp slt i32 %i, 4
      br i1 %i.cmp, label %body, label %reduction.exit



        body:
          %add = fadd double %phi, 4.2
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      br label %reduction.for

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


; CHECK: No modification has been made
; STATS: 1 polly-zone       - Number of not zone-analyzable arrays
