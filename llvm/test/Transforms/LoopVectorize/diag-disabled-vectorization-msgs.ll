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
; FORCEDONLY-NOT: LV: Not vectorizing: #pragma vectorize disable
; FORCEDONLY-NOT: LV: Not vectorizing: Disabled/already vectorized
; FORCEDONLY-NOT: LV: Not vectorizing: Cannot prove legality
; FORCEDONLY: LV: Loop hints: force=?
; FORCEDONLY: LV: Not vectorizing: VectorizeOnlyWhenForced is set, and no #pragma vectorize enable
; FORCEDONLY: remark:
; FORCEDONLY-SAME: loop not vectorized: only vectorizing loops that explicitly request it
; FORCEDONLY: LV: Loop hints prevent vectorization

define double @disabled_loop_vectorization(ptr %src) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %rdx = phi double [ 0.000000e+00, %entry ], [ %add, %loop ]
  %arrayidx = getelementptr inbounds nuw double, ptr %src, i64 %iv
  %1 = load double, ptr %arrayidx, align 8
  %sub = fsub fast double %1, 1.234e+0
  %mul = fmul fast double %sub, %sub
  %add = fadd fast double %mul, %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, 15
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !0

exit:
  ret double %add
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 false}
