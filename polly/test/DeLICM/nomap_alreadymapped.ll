; RUN: opt %loadNPMPolly '-passes=print<polly-delicm>' -disable-output < %s | FileCheck %s
;
;    void func(double *A) {
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        double phi1 = 0.0, phi2 = 0.0;
;        for (int i = 0; i < 4; i += 1) { /* reduction */
;          phi1 += 4.2;
;          phi2 += 29.0;
;        }
;        A[j] = phi1 + phi2;
;      }
;    }
;
; Check that we cannot map both, %phi1 and %phi2 to A[j] (conflict).
; Note that it is undefined which one will be mapped. We keep the test
; symmetric so it passes if either one is mapped.
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
      %phi1 = phi double [0.0, %reduction.preheader], [%add1, %reduction.inc]
      %phi2 = phi double [0.0, %reduction.preheader], [%add2, %reduction.inc]
      %i.cmp = icmp slt i32 %i, 4
      br i1 %i.cmp, label %body, label %reduction.exit



        body:
          %add1 = fadd double %phi1, 4.2
          %add2 = fadd double %phi2, 29.0
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      br label %reduction.for

    reduction.exit:
      %A_idx = getelementptr inbounds double, ptr %A, i32 %j
      %sum = fadd double %phi1, %phi2
      store double %sum, ptr %A_idx
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
