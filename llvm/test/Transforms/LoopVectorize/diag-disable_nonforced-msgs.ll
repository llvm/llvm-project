; This test checks that we emit only the correct debug messages and
; optimization remark when the loop vectorizer is disabled by loop metadata
; that requests no loop transformations.

; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -pass-remarks=loop-vectorize \
; RUN:     -pass-remarks-missed=loop-vectorize \
; RUN:     -pass-remarks-analysis=loop-vectorize -debug -disable-output \
; RUN:     -force-vector-interleave=1 -force-vector-width=2 \
; RUN:     < %s 2>&1 | FileCheck %s
; CHECK-NOT: LV: We can vectorize this loop
; CHECK-NOT: LV: Not vectorizing: #pragma vectorize disable.
; CHECK-NOT: LV: [BUG] Not vectorizing: loop vect disabled for an unknown reason
; CHECK-NOT: LV: Not vectorizing: VectorizeOnlyWhenForced is set
; CHECK-NOT: LV: Not vectorizing: Disabled/already vectorized
; CHECK-NOT: LV: Not vectorizing: Cannot prove legality
; CHECK: LV: Loop hints: force=disabled
; CHECK: LV: Not vectorizing: loop hasDisableAllTransformsHint.
; CHECK: remark:
; CHECK-SAME: loop not vectorized: loop transformations are disabled
; CHECK: LV: Loop hints prevent vectorization
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
define void @disable_nonforced(ptr nocapture %a, i32 %n) {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body, label %for.end

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = trunc i64 %indvars.iv to i32
  store i32 %0, ptr %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

!0 = !{!0, !{!"llvm.loop.disable_nonforced"}}
