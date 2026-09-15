; REQUIRES: asserts

; DEFINE: %{cmd} = opt -S -p loop-vectorize -debug-only=loop-vectorize --disable-output \
; DEFINE: -epilogue-tail-folding-policy=prefer-fold-tail -pass-remarks-analysis=loop-vectorize

; RUN: %{cmd} -force-vector-width=16 -epilogue-vectorization-force-VF=8 < %s 2>&1 | FileCheck %s

; RUN: %{cmd} -force-vector-width=16 -epilogue-vectorization-force-VF=8 -enable-epilogue-vectorization=false \
; RUN: < %s 2>&1 | FileCheck %s --check-prefix=CHECK-DISABLED-EPILOG

; RUN: %{cmd}  -epilogue-vectorization-force-VF=8 < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FORCED-MAIN-VF

; RUN: %{cmd} -force-vector-width=16 < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FORCED-EPILOGUE-VF

; RUN: %{cmd} -force-vector-width=8 -epilogue-vectorization-force-VF=8 < %s 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-VFs

; RUN: %{cmd} -force-vector-width=16 -epilogue-vectorization-force-VF=8 -enable-early-exit-vectorization-with-side-effects \
; RUN: < %s 2>&1 | FileCheck %s --check-prefix=CHECK-DISABLED-EARLY-EXIT

; RUN: %{cmd} -force-vector-width=16 -epilogue-vectorization-force-VF=8 -enable-vplan-native-path \
; RUN: < %s 2>&1 | FileCheck %s --check-prefix=CHECK-OUTER-LOOP


define void @test_epilogue_tf(ptr %A, i64 %n, i8 %val) {
; CHECK-LABEL: LV: Checking a loop in 'test_epilogue_tf'
; CHECK: LV: epilogue tail-folding is not supported yet
; CHECK: remark: <unknown>:0:0: The epilogue-tail-folding policy prefer-fold-tail is not supported yet, fall back to a normal epilogue
;
; CHECK-DISABLED-EPILOG-LABEL: LV: Checking a loop in 'test_epilogue_tf'
; CHECK-DISABLED-EPILOG: remark: <unknown>:0:0: Options conflict, epilogue vectorization is disallowed while epilogue tail-folding allowed!
;
; CHECK-NO-FORCED-MAIN-VF-LABEL: Checking a loop in 'test_epilogue_tf'
; CHECK-NO-FORCED-MAIN-VF: remark: <unknown>:0:0: For now, epilogue tail-folding can't be applied without forced main/epilogue loop VF

; CHECK-NO-FORCED-EPILOGUE-VF-LABEL: Checking a loop in 'test_epilogue_tf'
; CHECK-NO-FORCED-EPILOGUE-VF: remark: <unknown>:0:0: For now, epilogue tail-folding can't be applied without forced main/epilogue loop VF
;
; CHECK-INVALID-VFs-LABEL: Checking a loop in 'test_epilogue_tf'
; CHECK-INVALID-VFs: remark: <unknown>:0:0: For now, epilogue tail-folding can't be applied when VF of the main loop <= VF of the epilogue
;

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %A, i64 %iv
  store i8 %val, ptr %arrayidx, align 1
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp ne i64 %iv.next, %n
  br i1 %exitcond, label %for.body, label %exit

exit:
  ret void
}

define i16 @require_scalar_epilogue(ptr %dst, i64 %x) {
; CHECK-LABEL: Checking a loop in 'require_scalar_epilogue'
; CHECK: remark: <unknown>:0:0: Epilogue tail-folding can't be applied because scalar epilogue is required. Fall back to a normal epilogue
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

define i32 @opt_for_size(ptr %p, i32 %n, i8 %val) optsize {
; CHECK-LABEL: Checking a loop in 'opt_for_size'
; CHECK: remark: <unknown>:0:0: Not applying tail-folding to the epilogue, since no epilogue is allowed
;
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %iv
  store i8 %val, ptr %arrayidx, align 1
  %inc = add nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 0
}

define i32 @low_tc(ptr %p, i8 %val)  {
; CHECK-LABEL: Checking a loop in 'low_tc'
; CHECK: remark: <unknown>:0:0: Not applying tail-folding to the epilogue, since no epilogue is allowed.
;
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %p, i32 %iv
  store i8 %val, ptr %arrayidx, align 1
  %inc = add nsw i32 %iv, 1
  %exitcond = icmp eq i32 %inc, 8
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 0
}

define i1 @early_exit(ptr %A, i64 %n, i8 %find) {
; CHECK-DISABLED-EARLY-EXIT-LABEL: LV: Checking a loop in 'early_exit'
; CHECK-DISABLED-EARLY-EXIT: remark: <unknown>:0:0: Epilogue tail-folding is not supported yet for early-exit loops
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %cont ]
  %arrayidx = getelementptr inbounds i8, ptr %A, i64 %iv
  %val = load i8, ptr %arrayidx, align 1
  %exitcond = icmp eq i8 %val, %find
  br i1 %exitcond, label %exit, label %cont

cont:
  %iv.next = add nuw nsw i64 %iv, 1
  %contcond = icmp ne i64 %iv.next, %n
  br i1 %contcond, label %for.body, label %exit
exit:
  ret i1 %exitcond
}

; For this function, the check line is not related to epilogue tail-folding, but when vectorizing this case gets supported,
; the check line should be changed to: Epilogue tail-folding is not supported yet for early-exit loops, same as the case above.
define void @combined_exit_conditions(ptr align 4 dereferenceable(80) readonly %src, ptr align 4 dereferenceable(80) noalias %dst, ptr align 4 dereferenceable(80) readonly %pred) {
; CHECK-DISABLED-EARLY-EXIT-LABEL: LV: Checking a loop in 'combined_exit_conditions'
; CHECK-DISABLED-EARLY-EXIT: remark: <unknown>:0:0: loop not vectorized: Cannot vectorize uncountable loop
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %src.ptr = getelementptr inbounds nuw [4 x i8], ptr %src, i64 %iv
  %data = load i32, ptr %src.ptr, align 4
  %add = add nsw i32 %data, 1
  %dst.ptr = getelementptr inbounds nuw [4 x i8], ptr %dst, i64 %iv
  store i32 %add, ptr %dst.ptr, align 4
  %ee.ptr = getelementptr inbounds nuw [4 x i8], ptr %pred, i64 %iv
  %ee.val = load i32, ptr %ee.ptr, align 4
  %ee.cmp = icmp ne i32 %ee.val, 0
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cmp = icmp eq i64 %iv.next, 20
  %combined.cond = select i1 %ee.cmp, i1 true, i1 %counted.cmp
  br i1 %combined.cond, label %exit, label %for.body

exit:
  ret void
}

define void @test_outer_loop(ptr %A, i64 %m) {
; CHECK-OUTER-LOOP-LABEL: Checking a loop in 'test_outer_loop'
; CHECK-OUTER-LOOP: remark: <unknown>:0:0: Epilogue tail-folding is not supported for outer loop
;
entry:
  br label %outer.header

outer.header:
  %iv.outer = phi i64 [ 0, %entry ], [ %iv.outer.next, %outer.latch ]
  br label %inner

inner:
  %iv.inner = phi i64 [ 0, %outer.header ], [ %iv.inner.next, %inner ]
  %gep = getelementptr inbounds i8, ptr %A, i64 %iv.inner
  store i8 0, ptr %gep, align 1
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

!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.vectorize.enable"}

