; RUN: opt %loadNPMPolly '-passes=print<polly-delicm>' -disable-output < %s | FileCheck %s
;
;    void func(double *A) {
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        double phi = 0.0;
;        for (int i = 0; i < 4; i += 1) /* reduction */
;          if (phi < 0.0)
;            A[j] = undef;
;          phi += 4.2;
;        A[j] = phi;
;      }
;    }
;
; The MAY_WRITE in reduction.for.true conflict with a write of %phi to
; A[j] if %phi would be mapped to it. Being a MAY_WRITE avoids being target
; of a mapping itself.
;
; TODO: There is actually no reason why these conflict. The MAY_WRITE is an
; explicit write, defined to occur always before all implicit writes as the
; write of %phi would be. There is currently no differentiation between
; implicit and explicit writes in Polly.
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
      %A_idx = getelementptr inbounds double, ptr %A, i32 %j
      %phi.cmp = fcmp ogt double %phi, 0.0
      br i1 %phi.cmp, label %reduction.for.true, label %reduction.for.unconditional

    reduction.for.true:
       store double undef, ptr %A_idx
       br label %reduction.for.unconditional

    reduction.for.unconditional:
      %i.cmp = icmp slt i32 %i, 4
      br i1 %i.cmp, label %body, label %reduction.exit



        body:
          %add = fadd double %phi, 4.2
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      br label %reduction.for

    reduction.exit:
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
