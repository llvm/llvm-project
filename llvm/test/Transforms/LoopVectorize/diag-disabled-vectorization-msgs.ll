; TEST 1
; This test checks that we emit only the correct debug messages and
; optimization remark when the loop vectorizer is disabled by loop metadata.

; REQUIRES: asserts
; RUN: opt -S -passes=loop-vectorize -pass-remarks=loop-vectorize \
; RUN:     -pass-remarks-missed=loop-vectorize \
; RUN:     -pass-remarks-analysis=loop-vectorize -debug \
; RUN:     < %s 2>&1 | FileCheck --dump-input=always --check-prefix=METADATA %s
; METADATA-LABEL: 'disabled_loop_vectorization' from <stdin>
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
; RUN: opt -S -passes='loop-vectorize<vectorize-forced-only>' \
; RUN:   -pass-remarks=loop-vectorize \
; RUN:   -pass-remarks-missed=loop-vectorize \
; RUN:   -pass-remarks-analysis=loop-vectorize -debug \
; RUN:   2>&1 | FileCheck --dump-input=always --check-prefix=FORCEDONLY %s
; FORCEDONLY-LABEL: 'disabled_loop_vectorization' from <stdin>
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

; TEST 3
; This test checks that we emit only the correct debug messages and
; optimization remark when the loop vectorizer is disabled by loop metadata
; that requests no loop transformations.

; RUN: opt -S -passes=loop-vectorize -pass-remarks=loop-vectorize \
; RUN:     -pass-remarks-missed=loop-vectorize \
; RUN:     -pass-remarks-analysis=loop-vectorize -debug \
; RUN:     -force-vector-interleave=1 -force-vector-width=2 \
; RUN:     < %s 2>&1 | FileCheck --dump-input=always %s
; CHECK-LABEL: 'disable_nonforced' from <stdin>
; CHECK-NOT: LV: We can vectorize this loop
; CHECK-NOT: LV: Not vectorizing: #pragma vectorize disable.
; CHECK-NOT: LV: Not vectorizing: VectorizeOnlyWhenForced is set
; CHECK-NOT: LV: Not vectorizing: Disabled/already vectorized
; CHECK-NOT: LV: Not vectorizing: Cannot prove legality
; CHECK: LV: Loop hints: force=disabled
; CHECK: LV: Not vectorizing: loop hasDisableAllTransformsHint.
; CHECK: remark:
; CHECK-SAME: loop not vectorized: loop transformations are disabled
; CHECK: LV: Loop hints prevent vectorization
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
