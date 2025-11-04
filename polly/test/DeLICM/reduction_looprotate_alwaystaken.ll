; RUN: opt %loadPolly -polly-flatten-schedule -polly-delicm-overapproximate-writes=true -polly-delicm-compute-known=true -polly-print-delicm -disable-output < %s | FileCheck %s
;
; Verify that delicm can cope with never taken PHI incoming edges.
; The edge %body -> %body_phi is never taken, hence the access MemoryKind::PHI,
; WRITE in %body for %phi is never used.
; When mapping %phi, the write's access relation is the empty set.
;
;    void func(double *A) {
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        for (int i = 0; i < 4; i += 1) { /* reduction */
;          double phi = 21.0;
;          if (j < 10) // Tautology, since 0<=j<2
;            phi = 42.0;
;        }
;        A[j] = phi;
;      }
;    }
;
define void @func(ptr noalias nonnull %A, ptr noalias nonnull %dummy) {
entry:
  br label %outer.preheader

outer.preheader:
  br label %outer.for

outer.for:
  %j = phi i32 [0, %outer.preheader], [%j.inc, %outer.inc]
  %j.cmp = icmp slt i32 %j, 2
  br i1 %j.cmp, label %reduction.preheader, label %outer.exit


    reduction.preheader:
      %A_idx = getelementptr inbounds double, ptr %A, i32 %j
      br label %reduction.for

    reduction.for:
      %i = phi i32 [0, %reduction.preheader], [%i.inc, %reduction.inc]
      br label %body



        body:
          %cond = icmp slt i32 %j, 10
          br i1 %cond, label %alwaystaken, label %body_phi

        alwaystaken:
          store double 0.0, ptr %dummy
          br label %body_phi

        body_phi:
          %phi = phi double [21.0, %body], [42.0, %alwaystaken]
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      %i.cmp = icmp slt i32 %i.inc, 4
      br i1 %i.cmp, label %reduction.for, label %reduction.exit

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
; CHECK:     PHI scalars mapped:    1
; CHECK: }
