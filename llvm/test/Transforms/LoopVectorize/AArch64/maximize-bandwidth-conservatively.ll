; REQUIRES: asserts
; RUN: opt < %s -mtriple aarch64-linux-gnu -mattr=+sve -passes=loop-vectorize -vectorizer-maximize-bandwidth -S -debug-only=loop-vectorize 2>&1 | FileCheck %s
; RUN: opt < %s -mtriple aarch64-linux-gnu -mattr=+sve -passes=loop-vectorize -vectorizer-maximize-bandwidth -vectorizer-maximize-bandwidth-conservatively -S -debug-only=loop-vectorize 2>&1 | FileCheck %s --check-prefix=CHECK-CONS

define void @f(i32 %n, ptr noalias %a, ptr %b, ptr %c) {
; The following loop is an example where choosing a larger vector width reduces
; the number of instructions but may lead to performance degradation due to the
; FP pipeline becoming a bottleneck.
; 
; void f(int n, short *restrict a, long *b, double *c) {
;   for (int i = 0; i < n; i++) {
;     a[i] = b[i] + c[i];
;   }
; }

; In the usual cost model, vscale x 8 is chosen.
; CHECK: Cost for VF vscale x 2: 8 (Estimated cost per lane: 4.0)
; CHECK: Cost for VF vscale x 4: 14 (Estimated cost per lane: 3.5)
; CHECK: Cost for VF vscale x 8: 26 (Estimated cost per lane: 3.2)
; CHECK: LV: Selecting VF: vscale x 8.

; In a conservative cost model, a larger vector width is chosen only if it is
; superior when compared solely based on the cost of the FP pipeline, in
; addition to the usual model.
; CHECK-CONS: Cost for VF vscale x 2: 3 (Estimated cost per lane: 1.5)
; CHECK-CONS: Cost for VF vscale x 4: 7 (Estimated cost per lane: 1.8)
; CHECK-CONS: Cost for VF vscale x 8: 15 (Estimated cost per lane: 1.9)
; CHECK-CONS: LV: Selecting VF: vscale x 2.

entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds nuw i64, ptr %b, i64 %indvars.iv
  %0 = load i64, ptr %arrayidx, align 8
  %conv = sitofp i64 %0 to double
  %arrayidx2 = getelementptr inbounds nuw double, ptr %c, i64 %indvars.iv
  %1 = load double, ptr %arrayidx2, align 8
  %add = fadd double %1, %conv
  %conv3 = fptosi double %add to i16
  %arrayidx5 = getelementptr inbounds nuw i16, ptr %a, i64 %indvars.iv
  store i16 %conv3, ptr %arrayidx5, align 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}
