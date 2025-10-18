; TEST 1
; This test checks that we emit only the correct debug messages and
; optimization remark when the loop vectorizer is disabled by loop metadata.

; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -pass-remarks=loop-vectorize \
; RUN:     -pass-remarks-missed=loop-vectorize \
; RUN:     -pass-remarks-analysis=loop-vectorize -debug -disable-output \
; RUN:     < %s 2>&1 | FileCheck --check-prefix=METADATA %s
; METADATA-NOT: LV: We can vectorize this loop
; METADATA-NOT: LV: Not vectorizing: loop hasDisableAllTransformsHint
; METADATA-NOT: LV: [BUG] Not vectorizing: loop vect disabled for an unknown reason
; METADATA-NOT: LV: Not vectorizing: VectorizeOnlyWhenForced is set
; METADATA-NOT: LV: Not vectorizing: Disabled/already vectorized
; METADATA-NOT: LV: Not vectorizing: Cannot prove legality
; METADATA: LV: Loop hints: force=disabled
; METADATA: LV: Not vectorizing: #pragma vectorize disable.
; METADATA: remark:
; METADATA-SAME: loop not vectorized: vectorization is explicitly disabled
; METADATA: LV: Loop hints prevent vectorization

; TEST 2
; This test checks that we emit only the correct debug messages and
; optimization remark when the loop is not vectorized due to the 
; vectorize-forced-only pass option being set.

; Strip metadata for FORCEDONLY run, keep it for METADATA run
; RUN: sed 's/,[[:space:]]*!llvm\.loop[[:space:]]*!0//' %s | \
; RUN: opt -passes='loop-vectorize<vectorize-forced-only>' \
; RUN:   -pass-remarks=loop-vectorize \
; RUN:   -pass-remarks-missed=loop-vectorize \
; RUN:   -pass-remarks-analysis=loop-vectorize -debug -disable-output \
; RUN:   2>&1 | FileCheck --check-prefix=FORCEDONLY %s
; FORCEDONLY-NOT: LV: We can vectorize this loop
; FORCEDONLY-NOT: LV: Not vectorizing: loop hasDisableAllTransformsHint
; FORCEDONLY-NOT: LV: [BUG] Not vectorizing: loop vect disabled for an unknown reason
; FORCEDONLY-NOT: LV: Not vectorizing: #pragma vectorize disable
; FORCEDONLY-NOT: LV: Not vectorizing: Disabled/already vectorized
; FORCEDONLY-NOT: LV: Not vectorizing: Cannot prove legality
; FORCEDONLY: LV: Loop hints: force=?
; FORCEDONLY: LV: Not vectorizing: VectorizeOnlyWhenForced is set, and no #pragma vectorize enable
; FORCEDONLY: remark:
; FORCEDONLY-SAME: loop not vectorized: only vectorizing loops that explicitly request it
; FORCEDONLY: LV: Loop hints prevent vectorization

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
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 false}
