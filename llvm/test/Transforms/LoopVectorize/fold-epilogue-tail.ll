; REQUIRES: asserts

; RUN: opt -S -p loop-vectorize -debug-only=loop-vectorize --disable-output -epilogue-tail-folding-policy=prefer-fold-tail \
; RUN: -pass-remarks-analysis=loop-vectorize -force-vector-width=16 -epilogue-vectorization-force-VF=8 < %s 2>&1 | FileCheck %s

; RUN: opt -S -p loop-vectorize -debug-only=loop-vectorize --disable-output -epilogue-tail-folding-policy=prefer-fold-tail -force-vector-width=16 \
; RUN:  -pass-remarks-analysis=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FORCED-MAIN-VF

; RUN: opt -S -p loop-vectorize -debug-only=loop-vectorize --disable-output -epilogue-tail-folding-policy=prefer-fold-tail -epilogue-vectorization-force-VF=8  \
; RUN: -pass-remarks-analysis=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FORCED-EPILOGUE-VF

; RUN: opt -S -p loop-vectorize -debug-only=loop-vectorize -enable-epilogue-vectorization=false \
; RUN: --disable-output -force-vector-width=16 -epilogue-vectorization-force-VF=8 -epilogue-tail-folding-policy=prefer-fold-tail \
; RUN: -pass-remarks-analysis=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefix=CHECK-DISABLED-EPILOG


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

; This case can't be tail-folded because all the iterations will be executed by
; main vector loop.
define void @test_no_iterations_left(ptr %A) {
; CHECK-LABEL: Checking a loop in 'test_no_iterations_left'
; CHECK: LV: epilogue tail-folding is enabled
; CHECK: remark: <unknown>:0:0: This case of epilogue loop can't be tail-folded
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

define void @test_no_vf(ptr %A, i64 %n) {
; CHECK-NO-FORCED-MAIN-VF-LABEL: Checking a loop in 'test_no_vf'
; CHECK-NO-FORCED-MAIN-VF: remark: <unknown>:0:0: For now, Epilogue tail-folding can't be applied without forced epilogue/main loop VF
; CHECK-NO-FORCED-MAIN-VF-NOT: LV: epilogue tail-folding is enabled

; CHECK-NO-FORCED-EPILOGUE-VF-LABEL: Checking a loop in 'test_no_vf'
; CHECK-NO-FORCED-EPILOGUE-VF: remark: <unknown>:0:0: For now, Epilogue tail-folding can't be applied without forced epilogue/main loop VF
; CHECK-NO-FORCED-EPILOGUE-VF-NOT: LV: epilogue tail-folding is enabled
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
; CHECK-DISABLED-EPILOG-NOT: LV: epilogue tail-folding is enabled
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
; CHECK-NOT: LV: epilogue tail-folding is enabled
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
; CHECK-NOT: LV: epilogue tail-folding is enabled
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
