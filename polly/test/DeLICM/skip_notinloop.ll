; RUN: opt %loadNPMPolly '-passes=print<polly-delicm>' -pass-remarks-missed=polly-delicm -disable-output < %s 2>&1 | FileCheck %s
;
;    void func(double *A) {
;      double phi = 0.0;
;      for (int i = 0; i < 4; i += 1) /* reduction */
;        phi += 4.2;
;      A[0] = phi;
;    }
;
define void @func(ptr noalias nonnull %A) {
entry:
  br label %reduction.preheader

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
    store double %phi, ptr %A
    br label %return


return:
  ret void
}


; CHECK: skipped possible mapping target because it is not in a loop
