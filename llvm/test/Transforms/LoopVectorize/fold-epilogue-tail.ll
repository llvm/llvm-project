; REQUIRES: asserts

; RUN: opt -S -p loop-vectorize -debug-only=loop-vectorize --disable-output -epilogue-tail-folding-policy=prefer-fold-tail \
; RUN: -pass-remarks-analysis=loop-vectorize -force-vector-width=16 -epilogue-vectorization-force-VF=8 < %s 2>&1 | FileCheck %s

; RUN: opt -S -p loop-vectorize -debug-only=loop-vectorize -enable-vplan-native-path --disable-output \
; RUN: -epilogue-tail-folding-policy=prefer-fold-tail -pass-remarks-analysis=loop-vectorize \
; RUN: -force-vector-width=16 -epilogue-vectorization-force-VF=8 < %s 2>&1 | FileCheck %s --check-prefix=CHECK-OUTER-LOOP

; RUN: opt -S -p loop-vectorize -debug-only=loop-vectorize --disable-output -epilogue-tail-folding-policy=prefer-fold-tail -force-vector-width=16 \
; RUN:  -pass-remarks-analysis=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FORCED-MAIN-VF

; RUN: opt -S -p loop-vectorize -debug-only=loop-vectorize --disable-output -epilogue-tail-folding-policy=prefer-fold-tail -epilogue-vectorization-force-VF=8  \
; RUN: -pass-remarks-analysis=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FORCED-EPILOGUE-VF

; RUN: opt -S -p loop-vectorize -debug-only=loop-vectorize -enable-epilogue-vectorization=false \
; RUN: --disable-output -force-vector-width=16 -epilogue-vectorization-force-VF=8 -epilogue-tail-folding-policy=prefer-fold-tail \
; RUN: -pass-remarks-analysis=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefix=CHECK-DISABLED-EPILOG

; RUN: opt -S -p loop-vectorize -debug-only=loop-vectorize --disable-output -epilogue-tail-folding-policy=prefer-fold-tail \
; RUN: -force-vector-width=16 -epilogue-vectorization-force-VF=8 -vectorize-scev-check-threshold=0 < %s 2>&1 | FileCheck %s \
; RUN: --check-prefix=CHECK-NO-VPLANS

define void @test_epilogue_tf(ptr %A, i64 %n) {
; CHECK-LABEL: Checking a loop in 'test_epilogue_tf'
; CHECK: LV: epilogue tail-folding is enabled
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %A, i64 %iv
  store i8 1, ptr %arrayidx, align 1
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp ne i64 %iv.next, %n
  br i1 %exitcond, label %for.body, label %exit

exit:
  ret void
}

define void @test_outer_loop(ptr %A, i64 %m) {
; CHECK-OUTER-LOOP-LABEL: Checking a loop in 'test_outer_loop'
; CHECK-OUTER-LOOP: LV: Epilgue tail-folding is not supported for outer loop
;
entry:
  br label %outer.header

outer.header:
  %iv.outer = phi i64 [ 0, %entry ], [ %iv.outer.next, %outer.latch ]
  br label %inner

inner:
  %iv.inner = phi i64 [ 0, %outer.header ], [ %iv.inner.next, %inner ]
  %gep = getelementptr inbounds i32, ptr %A, i64 %iv.inner
  store i32 0, ptr %gep, align 4
  %iv.inner.next = add nuw nsw i64 %iv.inner, 1
  %inner.ec = icmp eq i64 %iv.inner.next, 8
  br i1 %inner.ec, label %outer.latch, label %inner

outer.latch:
  %iv.outer.next = add nuw nsw i64 %iv.outer, 1
  %outer.ec = icmp eq i64 %iv.outer.next, %m
  br i1 %outer.ec, label %exit, label %outer.header, !llvm.loop !1

exit:
  ret void
}

; This case can't be tail-folded because all the iterations will be executed by
; main vector loop.
define void @test_no_iterations_left(ptr %A) {
; CHECK-LABEL: Checking a loop in 'test_no_iterations_left'
; CHECK: LV: epilogue tail-folding is enabled
; CHECK: LV: This case of epilogue loop can't be tail-folded.
; CHECK: LV: Applying epilogue tail-folding failed, disable it.
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %A, i64 %iv
  store i8 1, ptr %arrayidx, align 1
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp ne i64 %iv.next, 64
  br i1 %exitcond, label %for.body, label %exit

exit:
  ret void
}

define void @test_no_fv(ptr %A, i64 %n) {
; CHECK-NO-FORCED-MAIN-VF-LABEL: Checking a loop in 'test_no_fv'
; CHECK-NO-FORCED-MAIN-VF: remark: <unknown>:0:0: For now, Epilogue tail-folding can't be applied without forced epilogue/main loop VF

; CHECK-NO-FORCED-EPILOGUE-VF-LABEL: Checking a loop in 'test_no_fv'
; CHECK-NO-FORCED-EPILOGUE-VF: remark: <unknown>:0:0: For now, Epilogue tail-folding can't be applied without forced epilogue/main loop VF
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %A, i64 %iv
  store i8 1, ptr %arrayidx, align 1
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp ne i64 %iv.next, %n
  br i1 %exitcond, label %for.body, label %exit

exit:
  ret void
}

define void @epilogue_is_disabled(ptr %a, i64 %n) {
; CHECK-DISABLED-EPILOG-LABEL: Checking a loop in 'epilogue_is_disabled'
; CHECK-DISABLED-EPILOG: remark: <unknown>:0:0: Options conflict, epilogue vectorization is disallowed while epilogue tail-folding allowed!
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, %n
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

define i16 @require_scalar_epilogue(ptr %dst, i64 %x) {
; CHECK-LABEL: Checking a loop in 'require_scalar_epilogue'
; CHECK: LV: Epilogue tail-folding can't be applied because scalar epilogue is required
; CHECK-NEXT: LV: Fall back to a normal epilogue
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i64 0, ptr %gep
  br label %loop.then

loop.then:
  %cmp3 = icmp ne i64 %iv, %x
  br i1 %cmp3, label %loop.latch, label %exit.1

loop.latch:
  %iv.next = add i64 %iv, 1
  br label %loop.header

exit.1:
  ret i16 0

exit.2:
  ret i16 1
}

define i32 @opt_for_size(ptr %p, i32 %n) optsize {
; CHECK-LABEL: Checking a loop in 'opt_for_size'
; CHECK: LV: No epilogue to apply tail-folding for.
; CHECK-NEXT: LV: Fall back to a normal epilogue
;
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %p, i32 %iv
  %0 = load i32, ptr %arrayidx, align 1
  %cmp1 = icmp eq i32 %0, 0
  %sel = select i1 %cmp1, i32 2, i32 1
  store i32 %sel, ptr %arrayidx, align 1
  %inc = add nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 0
}

; Can't build a valid vplan for this case because too many SCEV checks needed,
; more than the specfied limit.
define i64 @test_no_vplan_built(ptr %dst, i64 %n) {
; CHECK-NO-VPLANS-LABEL: Checking a loop in 'test_no_vplan_built'
; CHECK-NO-VPLANS: LV: epilogue tail-folding is enabled
; CHECK-NO-VPLANS: LV: no vplans have been built for main loop VF, bail out of epilogue tail-folding
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %dead.iv = phi i16 [ 0, %entry ], [ %dead.iv.next, %loop ]
  %prev = phi i64 [ 0, %entry ], [ %ext, %loop ]
  %iv.next = add nuw nsw i64 %iv, 1
  %dead.iv.next = add i16 %dead.iv, 1
  %ext = zext i16 %dead.iv.next to i64
  %gep = getelementptr inbounds i64, ptr %dst, i64 %prev
  store i64 %iv, ptr %gep, align 8
  %cmp = icmp slt i64 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  %result = phi i64 [ %ext, %loop ]
  ret i64 %result
}

!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.vectorize.enable"}
