; RUN: opt %loadNPMPolly '-passes=print<polly-delicm>' -pass-remarks-missed=polly-delicm -disable-output < %s 2>&1 | FileCheck %s
;
;    void func(double *A) {
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        double phi = 0.0;
;        for (int i = 0; i < 4; i += 1) /* reduction */
;          phi += 4.2;
;          if (phi > 42.0)
;            (void)A[j];
;          else
;            A[j] = 0.0;
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
  %j = phi i32 [0, %outer.preheader], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, 2
  br i1 %j.cmp, label %reduction.preheader, label %outer.exit


    reduction.preheader:
      br label %reduction.for

    reduction.for:
      %i = phi i32 [0, %reduction.preheader], [%i.inc, %reduction.inc]
      %phi = phi double [0.0, %reduction.preheader], [%add, %reduction.inc]
      %i.cmp = icmp slt i32 %i, 4
      br i1 %i.cmp, label %body, label %reduction.exit



        body:
          %add = fadd double %phi, 4.2
          %A_idxp = getelementptr inbounds double, ptr %A, i32 %j
          %add.cmp = fcmp ogt double %add, 42.0
          br i1 %add.cmp , label %body_true, label %body_false

        body_true:
          %dummy = load double, ptr %A_idxp
          br label %reduction.inc

        body_false:
          store double 0.0, ptr %A_idxp
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      br label %reduction.for

    reduction.exit:
      %A_idx = getelementptr inbounds double, ptr %A, i32 %j
      store double %phi, ptr %A_idx
      br label %outer.inc



outer.inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %outer.for

outer.exit:
  br label %return

return:
  ret void
}


; CHECK: store is in a non-affine subregion
