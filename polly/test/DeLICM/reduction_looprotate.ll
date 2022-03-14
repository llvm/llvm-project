; RUN: opt %loadPolly -polly-flatten-schedule -polly-print-delicm -disable-output < %s | FileCheck %s
;
;    void func(double *A) {
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        for (int i = 0; i < 4; i += 1) { /* reduction */
;          double phi = A[j];
;          phi += 4.2;
;          A[j] = phi;
;        }
;      }
;    }
;
; There is nothing to do in this case. All accesses are in %body.
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
      br label %reduction.for

    reduction.for:
      %i = phi i32 [0, %reduction.preheader], [%i.inc, %reduction.inc]
      br label %body



        body:
          %A_idx = getelementptr inbounds double, double* %A, i32 %j
          %val = load double, double* %A_idx
          %add = fadd double %val, 4.2
          store double %add, double* %A_idx
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      %i.cmp = icmp slt i32 %i.inc, 4
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


; CHECK: No modification has been made
