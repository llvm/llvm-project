; This test checks that we emit only the correct debug messages and
; optimization remark when the loop is not vectorized due to the 
; vectorize-forced-only pass option being set.

; REQUIRES: asserts
; RUN: opt -passes='loop-vectorize<vectorize-forced-only>' \
; RUN:   -pass-remarks=loop-vectorize \
; RUN:   -pass-remarks-missed=loop-vectorize \
; RUN:   -pass-remarks-analysis=loop-vectorize -debug -disable-output \
; RUN:   < %s 2>&1 | FileCheck %s
; CHECK-NOT: LV: We can vectorize this loop
; CHECK-NOT: LV: Not vectorizing: loop hasDisableAllTransformsHint
; CHECK-NOT: LV: [BUG] Not vectorizing: loop vect disabled for an unknown reason
; CHECK-NOT: LV: Not vectorizing: #pragma vectorize disable
; CHECK-NOT: LV: Not vectorizing: Disabled/already vectorized
; CHECK-NOT: LV: Not vectorizing: Cannot prove legality
; CHECK: LV: Loop hints: force=?
; CHECK: LV: Not vectorizing: VectorizeOnlyWhenForced is set, and no #pragma vectorize enable
; CHECK: remark:
; CHECK-SAME: loop not vectorized: only vectorizing loops that explicitly request it
; CHECK: LV: Loop hints prevent vectorization

define double @CompareDistmats(ptr %distmat1, ptr %distmat2){
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %add.lcssa = phi double [ %add, %for.body ]
  %div = fmul fast double %add.lcssa, 0x3FB1111111111111
  %0 = tail call fast double @llvm.sqrt.f64(double %div)
  ret double %0

for.body:                                         ; preds = %entry, %for.body
  %i.014 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %RMSD.013 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds nuw double, ptr %distmat1, i64 %i.014
  %1 = load double, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds nuw double, ptr %distmat2, i64 %i.014
  %2 = load double, ptr %arrayidx1, align 8
  %sub = fsub fast double %1, %2
  %mul = fmul fast double %sub, %sub
  %add = fadd fast double %mul, %RMSD.013
  %inc = add nuw nsw i64 %i.014, 1
  %exitcond.not = icmp eq i64 %inc, 15
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
