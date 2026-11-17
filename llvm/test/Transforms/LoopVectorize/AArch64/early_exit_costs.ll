; REQUIRES: asserts
; RUN: opt -S < %s -p loop-vectorize -disable-output \
; RUN: -debug-only=loop-vectorize 2>&1 | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

declare void @init_mem(ptr, i64);

define i64 @same_exit_block_pre_inc_use1_sve() #1 {
; CHECK-LABEL: LV: Checking a loop in 'same_exit_block_pre_inc_use1_sve'
; CHECK: LV: Selecting VF: vscale x 16
; CHECK: Calculating cost of work in exit block vector.early.exit
; CHECK-NEXT: Cost of 4 for VF vscale x 16: EMIT vp<{{.*}}> = first-active-lane vp<{{.*}}>
; CHECK-NEXT: Cost of 0 for VF vscale x 16: EMIT vp<{{.*}}> = add
; CHECK-NEXT: Cost of 1 for VF vscale x 16: vp<{{.*}}> = DERIVED-IV
; CHECK-NEXT: Cost of 2 for VF vscale x 16: vp<{{.*}}> = DERIVED-IV
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
; CHECK-NEXT: Cost of 7 for VF 16: EMIT vp<{{.*}}> = first-active-lane vp<{{.*}}>
; CHECK-NEXT: Cost of 0 for VF 16: EMIT vp<{{.*}}> = add
; CHECK-NEXT: Cost of 1 for VF 16: vp<{{.*}}> = DERIVED-IV
; CHECK-NEXT: Cost of 2 for VF 16: vp<{{.*}}> = DERIVED-IV
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

; Both the vector.early.exit block (cost 11) and the middle block (cost 5)
; extract the live-outs.
; FIXME: The vector loop is left through only one of the two blocks, so
; Cost = max(11, 5) suffices instead of Cost = 11 + 5.
define i64 @live_outs_in_early_exit_and_middle_block(ptr align 8 dereferenceable(1024) %p1, ptr align 8 dereferenceable(1024) %p2) {
; CHECK-LABEL: LV: Checking a loop in 'live_outs_in_early_exit_and_middle_block'
; CHECK: LV: Scalar loop costs: 7.
; CHECK: Cost for VF 2: 7 (
; CHECK: LV: Selecting VF: 2
; CHECK: Calculating cost of work in exit block vector.early.exit
; CHECK-NEXT: Cost of 7 for VF 2: EMIT vp<{{.*}}> = first-active-lane vp<{{.*}}>
; CHECK-NEXT: Cost of 2 for VF 2: EMIT {{.*}} = extract-lane vp<{{.*}}>, ir<%ld1>
; CHECK-NEXT: Cost of 2 for VF 2: EMIT {{.*}} = extract-lane vp<{{.*}}>, ir<%ld2>
; CHECK-NEXT: Cost of 0 for VF 2: EMIT vp<{{.*}}> = extract-last-part ir<%ld1>
; CHECK-NEXT: Cost of 2 for VF 2: EMIT vp<{{.*}}> = extract-last-lane vp<{{.*}}>
; CHECK-NEXT: Cost of 0 for VF 2: EMIT vp<{{.*}}> = extract-last-part ir<%ld2>
; CHECK-NEXT: Cost of 2 for VF 2: EMIT vp<{{.*}}> = extract-last-lane vp<{{.*}}>
; CHECK-NEXT: Cost of 1 for VF 2: EMIT vp<{{.*}}> = icmp eq ir<128>, vp<{{.*}}>
; CHECK-NEXT: Cost of 0 for VF 2: EMIT branch-on-cond vp<{{.*}}>
; CHECK-NEXT: LV: Minimum required TC for runtime checks to be profitable:6
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.p1 = getelementptr inbounds i64, ptr %p1, i64 %iv
  %ld1 = load i64, ptr %gep.p1, align 8
  %gep.p2 = getelementptr inbounds i64, ptr %p2, i64 %iv
  %ld2 = load i64, ptr %gep.p2, align 8
  %cmp = icmp eq i64 %ld1, %ld2
  br i1 %cmp, label %loop.latch, label %exit

loop.latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 128
  br i1 %ec, label %exit, label %loop.header

exit:
  %r1 = phi i64 [ %ld1, %loop.header ], [ %ld1, %loop.latch ]
  %r2 = phi i64 [ %ld2, %loop.header ], [ %ld2, %loop.latch ]
  %res = add i64 %r1, %r2
  ret i64 %res
}

; Both vector.early.exit blocks (cost 6 each) extract the live-outs and the
; middle block costs 1.
; FIXME: Cost = max(6, 6, 1) suffices, instead of Cost = 6 + 6 + 1.
define i64 @live_outs_in_multiple_early_exits(ptr align 8 dereferenceable(1024) %p1, ptr align 8 dereferenceable(1024) %p2, ptr align 8 dereferenceable(1024) %p3) {
; CHECK-LABEL: LV: Checking a loop in 'live_outs_in_multiple_early_exits'
; CHECK: LV: Scalar loop costs: 10.
; CHECK: Cost for VF 2: 9 (
; CHECK: LV: Selecting VF: 2
; CHECK: Calculating cost of work in exit block vector.early.exit.0
; CHECK-NEXT: Cost of 2 for VF 2: EMIT {{.*}} = extract-lane vp<{{.*}}>, ir<%ld1>
; CHECK-NEXT: Cost of 2 for VF 2: EMIT {{.*}} = extract-lane vp<{{.*}}>, ir<%ld2>
; CHECK-NEXT: Cost of 2 for VF 2: EMIT {{.*}} = extract-lane vp<{{.*}}>, ir<%ld3>
; CHECK-NEXT: Calculating cost of work in exit block vector.early.exit.1
; CHECK-NEXT: Cost of 2 for VF 2: EMIT {{.*}} = extract-lane vp<{{.*}}>, ir<%ld1>
; CHECK-NEXT: Cost of 2 for VF 2: EMIT {{.*}} = extract-lane vp<{{.*}}>, ir<%ld2>
; CHECK-NEXT: Cost of 2 for VF 2: EMIT {{.*}} = extract-lane vp<{{.*}}>, ir<%ld3>
; CHECK-NEXT: Cost of 1 for VF 2: EMIT vp<{{.*}}> = icmp eq ir<128>, vp<{{.*}}>
; CHECK-NEXT: Cost of 0 for VF 2: EMIT branch-on-cond vp<{{.*}}>
; CHECK-NEXT: LV: Minimum required TC for runtime checks to be profitable:4
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.p1 = getelementptr inbounds i64, ptr %p1, i64 %iv
  %ld1 = load i64, ptr %gep.p1, align 8
  %gep.p2 = getelementptr inbounds i64, ptr %p2, i64 %iv
  %ld2 = load i64, ptr %gep.p2, align 8
  %gep.p3 = getelementptr inbounds i64, ptr %p3, i64 %iv
  %ld3 = load i64, ptr %gep.p3, align 8
  %c1 = icmp eq i64 %ld1, 0
  br i1 %c1, label %exit.1, label %loop.body

loop.body:
  %c2 = icmp eq i64 %ld2, 0
  br i1 %c2, label %exit.2, label %loop.latch

loop.latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 128
  br i1 %ec, label %exit.latch, label %loop.header

exit.1:
  %add.1 = add i64 %ld1, %ld2
  %res.1 = add i64 %add.1, %ld3
  ret i64 %res.1

exit.2:
  %add.2 = add i64 %ld1, %ld2
  %res.2 = add i64 %add.2, %ld3
  ret i64 %res.2

exit.latch:
  ret i64 0
}

define i64 @vectorization_not_profitable_due_to_trunc(ptr dereferenceable(800) %src) {
; CHECK-LABEL: LV: Checking a loop in 'vectorization_not_profitable_due_to_trunc'
; CHECK: LV: Selecting VF: 1.
; CHECK: LV: Vectorization is possible but not beneficial.
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.src = getelementptr inbounds i64, ptr %src, i64 %iv
  %l = load i64, ptr %gep.src, align 1
  %t = trunc i64 %l to i1
  br i1 %t, label %exit.0, label %loop.latch, !prof !0

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

!0 = !{!"branch_weights", i32 1, i32 1}

attributes #1 = { "target-features"="+sve" vscale_range(1,16) }
