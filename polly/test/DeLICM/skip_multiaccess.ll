; RUN: opt %loadPolly -polly-delicm -pass-remarks-missed=polly-delicm -disable-output < %s 2>&1 | FileCheck %s
;
; llvm.org/PR34485
; llvm.org/PR34989
;
; The memset causes the array A to be divided into i8-sized subelements.
; The the regular store then writes multiple of these subelements, which
; we do not support currently.
;
;    void func(double *A) {
;      memset(A, 0, 4);
;      for (int j = 0; j < 2; j += 1) { /* outer */
;        double phi = 0.0;
;        for (int i = 0; i < 4; i += 1) /* reduction */
;          phi += 4.2;
;        A[j] = phi;
;      }
;    }

declare void @llvm.memset.p0f64.i64(double* nocapture, i8, i64, i32, i1)

define void @func(double* noalias nonnull %A) {
entry:
  br label %outer.for

outer.for:
  %j = phi i32 [0, %entry], [%j.inc, %outer.inc]
  call void @llvm.memset.p0f64.i64(double* %A, i8 0, i64 4, i32 1, i1 false)
  %j.cmp = icmp slt i32 %j, 2
  br i1 %j.cmp, label %reduction.for, label %outer.exit


    reduction.for:
      %i = phi i32 [0, %outer.for], [%i.inc, %reduction.inc]
      %phi = phi double [0.0, %outer.for], [%add, %reduction.inc]
      %i.cmp = icmp slt i32 %i, 4
      br i1 %i.cmp, label %body, label %reduction.exit



        body:
          %add = fadd double %phi, 4.2
          br label %reduction.inc



    reduction.inc:
      %i.inc = add nuw nsw i32 %i, 1
      br label %reduction.for

    reduction.exit:
      %A_idx = getelementptr inbounds double, double* %A, i32 %j
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


; CHECK: skipped possible mapping target because it writes more than one element
