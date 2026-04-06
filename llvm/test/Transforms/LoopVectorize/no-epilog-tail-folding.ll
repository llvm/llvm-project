; RUN: opt -S < %s -p loop-vectorize -debug-only=loop-vectorize --disable-output \
; RUN: -tail-folding-policy=predicated-epilogue 2>&1 | FileCheck %s

; RUN: opt -S < %s -p loop-vectorize -debug-only=loop-vectorize -enable-epilogue-vectorization=false \
; RUN: --disable-output -tail-folding-policy=predicated-epilogue 2>&1 \
; RUN: | FileCheck %s --check-prefix=CHECK-DISABLED-EPILOG

; RUN: opt -S < %s -p loop-vectorize -debug-only=loop-vectorize -enable-interleaved-mem-accesses=true \
; RUN: --disable-output -tail-folding-policy=predicated-epilogue 2>&1 \
; RUN: | FileCheck %s --check-prefix=CHECK-INTERLEAVE

target datalayout = "E-m:e-p:32:32-i64:32-f64:32:64-a:0:32-n32-S128"

define void @test_epilogue_tf(ptr %a, i64 %n) {
; CHECK: LV: Checking a loop in 'test_epilogue_tf'
; CHECK: LV: Epilogue tail folding is enabled
entry:
  %cmp1 = icmp sgt i64 %n, 0
  br i1 %cmp1, label %for.body, label %for.end
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

define void @epilogue_is_disabled(ptr %a, i64 %n) {
; CHECK-DISABLED-EPILOG: LV: Checking a loop in 'epilogue_is_disabled'
; CHECK-DISABLED-EPILOG: LV: Options conflict, epilogue vectorization is disallowed while epilogue predication allowed!
; CHECK-DISABLED-EPILOG-NEXT: LV: Disallow epilogue predication
entry:
  %cmp1 = icmp sgt i64 %n, 0
  br i1 %cmp1, label %for.body, label %for.end
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

@AB = common global [1024 x i32] zeroinitializer, align 4
@CD = common global [1024 x i32] zeroinitializer, align 4
define void @interleave_requires_scalar_epilog(i32 %C, i32 %D) {
; CHECK-INTERLEAVE: LV: Checking a loop in 'interleave_requires_scalar_epilog'
; CHECK-INTERLEAVE: LV: Epilogue tail folding can't be applied because scalar epilogue is required
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx0 = getelementptr inbounds [1024 x i32], ptr @AB, i64 0, i64 %indvars.iv
  %tmp = load i32, ptr %arrayidx0, align 4
  %tmp1 = or disjoint i64 %indvars.iv, 1
  %arrayidx1 = getelementptr inbounds [1024 x i32], ptr @AB, i64 0, i64 %tmp1
  %tmp2 = load i32, ptr %arrayidx1, align 4
  %add = add nsw i32 %tmp, %C
  %mul = mul nsw i32 %tmp2, %D
  %arrayidx2 = getelementptr inbounds [1024 x i32], ptr @CD, i64 0, i64 %indvars.iv
  store i32 %add, ptr %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [1024 x i32], ptr @CD, i64 0, i64 %tmp1
  store i32 %mul, ptr %arrayidx3, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp slt i64 %indvars.iv.next, 1024
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

define i16 @early_exit_requires_scalar_epilog(ptr %dst, i64 %x) {
; CHECK: LV: Checking a loop in 'early_exit_requires_scalar_epilog'
; CHECK: LV: Epilogue tail folding can't be applied because scalar epilogue is required
entry:
  br label %loop.header
loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i64 0, ptr %gep
  br i1 true, label %loop.then, label %exit.2
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
; CHECK: LV: Checking a loop in 'opt_for_size'
; CHECK: LV: No epilogue to apply tail folding for.
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

define i32 @max_reduction_epilog(ptr %src, i64 %N) {
; CHECK: LV: Checking a loop in 'max_reduction_epilog'
; CHECK: LV: Epilogue tail folding is not supported for select-cmp Reductions
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %red = phi i32 [ 0, %entry ], [ %select, %loop ]
  %gep = getelementptr inbounds i32, ptr %src, i64 %iv
  %load = load i32, ptr %gep, align 1
  %icmp = icmp ugt i32 %load, %red
  %select = select i1 %icmp, i32 %load, i32 %red
  %iv.next = add i64 %iv, 1
  %icmp3 = icmp eq i64 %iv, %N
  br i1 %icmp3, label %exit, label %loop

exit:
  ret i32 %select
}
