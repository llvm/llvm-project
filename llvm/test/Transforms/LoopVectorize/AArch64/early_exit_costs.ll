; REQUIRES: asserts
; RUN: opt -S < %s -p loop-vectorize -disable-output \
; RUN:   -debug-only=loop-vectorize 2>&1 | FileCheck %s --check-prefixes=CHECK

target triple = "aarch64-unknown-linux-gnu"

declare void @init_mem(ptr, i64);

define i64 @same_exit_block_pre_inc_use1_sve() #1 {
; CHECK-LABEL: LV: Checking a loop in 'same_exit_block_pre_inc_use1_sve'
; CHECK: LV: Selecting VF: vscale x 16
; CHECK: Calculating cost of work in exit block vector.early.exit
; CHECK-NEXT: Cost of 4 for VF vscale x 16: EMIT vp<{{.*}}> = first-active-lane ir<%cmp3>
; CHECK-NEXT: Cost of 0 for VF vscale x 16: EMIT vp<{{.*}}> = add
; CHECK-NEXT: Cost of 0 for VF vscale x 16: vp<{{.*}}> = DERIVED-IV
; CHECK-NEXT: Cost of 4 for VF vscale x 16: EMIT vp<{{.*}}> = first-active-lane ir<%cmp3>
; CHECK-NEXT: Cost of 0 for VF vscale x 16: EMIT vp<{{.*}}> = add
; CHECK-NEXT: Cost of 0 for VF vscale x 16: vp<{{.*}}> = DERIVED-IV
; CHECK: LV: Minimum required TC for runtime checks to be profitable:16
entry:
  %p1 = alloca [1024 x i8]
  %p2 = alloca [1024 x i8]
  call void @init_mem(ptr %p1, i64 1024)
  call void @init_mem(ptr %p2, i64 1024)
  br label %loop

loop:
  %index = phi i64 [ %index.next, %loop.inc ], [ 3, %entry ]
  %index2 = phi i64 [ %index2.next, %loop.inc ], [ 15, %entry ]
  %arrayidx = getelementptr inbounds i8, ptr %p1, i64 %index
  %ld1 = load i8, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, ptr %p2, i64 %index
  %ld2 = load i8, ptr %arrayidx1, align 1
  %cmp3 = icmp eq i8 %ld1, %ld2
  br i1 %cmp3, label %loop.inc, label %loop.end

loop.inc:
  %index.next = add i64 %index, 1
  %index2.next = add i64 %index2, 2
  %exitcond = icmp ne i64 %index.next, 67
  br i1 %exitcond, label %loop, label %loop.end

loop.end:
  %val1 = phi i64 [ %index, %loop ], [ 67, %loop.inc ]
  %val2 = phi i64 [ %index2, %loop ], [ 98, %loop.inc ]
  %retval = add i64 %val1, %val2
  ret i64 %retval
}

define i64 @same_exit_block_pre_inc_use1_nosve() {
; CHECK-LABEL: LV: Checking a loop in 'same_exit_block_pre_inc_use1_nosve'
; CHECK: LV: Selecting VF: 16
; CHECK: Calculating cost of work in exit block vector.early.exit
; CHECK-NEXT: Cost of 48 for VF 16: EMIT vp<{{.*}}> = first-active-lane ir<%cmp3>
; CHECK-NEXT: Cost of 0 for VF 16: EMIT vp<{{.*}}> = add
; CHECK-NEXT: Cost of 0 for VF 16: vp<{{.*}}> = DERIVED-IV
; CHECK-NEXT: Cost of 48 for VF 16: EMIT vp<{{.*}}> = first-active-lane ir<%cmp3>
; CHECK-NEXT: Cost of 0 for VF 16: EMIT vp<{{.*}}> = add
; CHECK-NEXT: Cost of 0 for VF 16: vp<{{.*}}> = DERIVED-IV
; CHECK: LV: Minimum required TC for runtime checks to be profitable:160
; CHECK-NEXT: LV: Vectorization is not beneficial: expected trip count < minimum profitable VF (64 < 160)
; CHECK-NEXT: LV: Too many memory checks needed.
entry:
  %p1 = alloca [1024 x i8]
  %p2 = alloca [1024 x i8]
  call void @init_mem(ptr %p1, i64 1024)
  call void @init_mem(ptr %p2, i64 1024)
  br label %loop

loop:
  %index = phi i64 [ %index.next, %loop.inc ], [ 3, %entry ]
  %index2 = phi i64 [ %index2.next, %loop.inc ], [ 15, %entry ]
  %arrayidx = getelementptr inbounds i8, ptr %p1, i64 %index
  %ld1 = load i8, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, ptr %p2, i64 %index
  %ld2 = load i8, ptr %arrayidx1, align 1
  %cmp3 = icmp eq i8 %ld1, %ld2
  br i1 %cmp3, label %loop.inc, label %loop.end

loop.inc:
  %index.next = add i64 %index, 1
  %index2.next = add i64 %index2, 2
  %exitcond = icmp ne i64 %index.next, 67
  br i1 %exitcond, label %loop, label %loop.end

loop.end:
  %val1 = phi i64 [ %index, %loop ], [ 67, %loop.inc ]
  %val2 = phi i64 [ %index2, %loop ], [ 98, %loop.inc ]
  %retval = add i64 %val1, %val2
  ret i64 %retval
}

define i64 @vectorization_not_profitable_due_to_trunc(ptr dereferenceable(800) %src) {
; CHECK-LABEL: LV: Checking a loop in 'vectorization_not_profitable_due_to_trunc'
; CHECK: LV: Selecting VF: 1.
; CHECK-NEXT: Calculating cost of work in exit block vector.early.exit:
; CHECK-NEXT: Cost of 1 for VF 1: EMIT vp<%first.active.lane> = first-active-lane ir<%t>
; CHECK-NEXT: Cost of 0 for VF 1: EMIT vp<%early.exit.value> = extract-lane vp<%first.active.lane>, ir<%l>
; CHECK-NEXT: LV: Vectorization is possible but not beneficial.
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.src = getelementptr inbounds i64, ptr %src, i64 %iv
  %l = load i64, ptr %gep.src, align 1
  %t = trunc i64 %l to i1
  br i1 %t, label %exit.0, label %loop.latch

loop.latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 100
  br i1 %ec, label %exit.1, label %loop.header

exit.0:
  %res = phi i64 [ %l, %loop.header ]
  ret i64 %res

exit.1:
  ret i64 0
}

attributes #1 = { "target-features"="+sve" vscale_range(1,16) }
