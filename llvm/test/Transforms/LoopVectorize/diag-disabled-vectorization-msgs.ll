; REQUIRES: asserts

; TEST 1
; Checks that we emit only the correct debug messages and
; optimization remark when the loop vectorizer is disabled by loop metadata.
; RUN: opt -S -passes=loop-vectorize -pass-remarks=loop-vectorize \
; RUN:     -pass-remarks-missed=loop-vectorize \
; RUN:     -pass-remarks-analysis=loop-vectorize -debug \
; RUN:     < %s 2>&1 | FileCheck --check-prefixes=METADATA,ALL %s
; TEST 2
; Checks that we emit only the correct debug messages and
; optimization remark when the loop is not vectorized due to the
; vectorize-forced-only pass option being set.
; Strip metadata for FORCEDONLY run, keep it for METADATA run
; RUN: sed 's/,[[:space:]]*!llvm\.loop[[:space:]]*!0//' %s | \
; RUN: opt -S -passes='loop-vectorize<vectorize-forced-only>' \
; RUN:   -pass-remarks=loop-vectorize \
; RUN:   -pass-remarks-missed=loop-vectorize \
; RUN:   -pass-remarks-analysis=loop-vectorize -debug \
; RUN:   2>&1 | FileCheck --check-prefixes=FORCEDONLY,ALL %s
; TEST 3
; Checks that we emit only the correct debug messages and
; optimization remark when the loop vectorizer is disabled by loop metadata
; that requests no loop transformations.
; RUN: opt -S -passes=loop-vectorize -pass-remarks=loop-vectorize \
; RUN:     -pass-remarks-missed=loop-vectorize \
; RUN:     -pass-remarks-analysis=loop-vectorize -debug \
; RUN:     -force-vector-interleave=1 -force-vector-width=2 \
; RUN:     < %s 2>&1 | FileCheck --check-prefix=ALL %s

; ALL-LABEL: 'disabled_loop_vectorization' from <stdin>
; ALL-NOT: LV: We can vectorize this loop
; ALL-NOT: LV: Not vectorizing: loop hasDisableAllTransformsHint
; ALL-NOT: LV: Not vectorizing: Disabled/already vectorized
; ALL-NOT: LV: Not vectorizing: Cannot prove legality
;
; METADATA-NOT: LV: Not vectorizing: VectorizeOnlyWhenForced is set
; METADATA: LV: Loop hints: force=disabled
; METADATA: LV: Not vectorizing: #pragma vectorize disable.
; METADATA: remark:
; METADATA-SAME: loop not vectorized: vectorization is explicitly disabled
;
; FORCEDONLY-NOT: LV: Not vectorizing: #pragma vectorize disable
; FORCEDONLY: LV: Loop hints: force=?
; FORCEDONLY: LV: Not vectorizing: VectorizeOnlyWhenForced is set, and no #pragma vectorize enable
; FORCEDONLY: remark:
; FORCEDONLY-SAME: loop not vectorized: only vectorizing loops that explicitly request it
;
; ALL: LV: Loop hints prevent vectorization
define void @disabled_loop_vectorization(ptr %src) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %arrayidx = getelementptr inbounds nuw double, ptr %src, i64 %iv
  store double 0.0, ptr %arrayidx, align 8
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, 15
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 false}

; ALL-LABEL: 'disable_nonforced' from <stdin>
; ALL-NOT: LV: We can vectorize this loop
; ALL-NOT: LV: Not vectorizing: #pragma vectorize disable.
; ALL-NOT: LV: Not vectorizing: VectorizeOnlyWhenForced is set
; ALL-NOT: LV: Not vectorizing: Disabled/already vectorized
; ALL-NOT: LV: Not vectorizing: Cannot prove legality
; ALL: LV: Loop hints: force=disabled
; ALL: LV: Not vectorizing: loop hasDisableAllTransformsHint.
; ALL: remark:
; ALL-SAME: loop not vectorized: loop transformations are disabled
; ALL: LV: Loop hints prevent vectorization
define void @disable_nonforced(ptr nocapture %a, i32 %n) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i32 %iv
  store i32 %iv, ptr %arrayidx, align 4
  %iv.next = add i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %end, label %loop, !llvm.loop !2

end:
  ret void
}
!2 = !{!2, !{!"llvm.loop.disable_nonforced"}}
