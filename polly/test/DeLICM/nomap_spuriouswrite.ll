; RUN: opt %loadNPMPolly '-passes=print<polly-delicm>' -disable-output < %s | FileCheck %s
;
;    void func(double *A) {
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        double phi = 0.0;
;        for (int i = 0; i < 4; i += 1) /* reduction */
;          phi += 4.2;
;        if (phi < 0.0)
;          A[j] = undef;
;        A[j] = phi;
;      }
;    }
;
; The MAY_WRITE in reduction.exit.true avoids that anything can be mapped to
; A[j] because it would be overwritten by that MAY_WRITE just before the final
; MUST_WRITE. Also nothing can be map to the MAY_WRITE itself because it is a
; MAY_WRITE.
;
define void @func(ptr noalias nonnull %A) {
entry:
  %fsomeval = fadd double 21.0, 21.0
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
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      br label %reduction.for

    reduction.exit:
      %A_idx = getelementptr inbounds double, ptr %A, i32 %j
      %phi.cmp = fcmp ogt double %phi, 0.0
      br i1 %phi.cmp, label %reduction.exit.true, label %reduction.exit.unconditional

    reduction.exit.true:
       store double undef, ptr %A_idx
       br label %reduction.exit.unconditional

    reduction.exit.unconditional:
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


; CHECK: Statistics {
; CHECK:     Compatible overwrites: 1
; CHECK: }
; CHECK: No modification has been made
