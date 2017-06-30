; RUN: opt < %s -loop-unroll -unroll-runtime=true -unroll-runtime-epilog=true -unroll-runtime-multi-exit=true -verify-dom-info -verify-loop-info -instcombine -S| FileCheck %s
; RUN: opt < %s -loop-unroll -unroll-runtime -unroll-count=2 -unroll-runtime-epilog=true -unroll-runtime-multi-exit=true -verify-dom-info -verify-loop-info -instcombine

; the second RUN generates an epilog remainder block for all the test
; cases below (it does not generate a loop).

; test with three exiting and three exit blocks.
; none of the exit blocks have successors
define void @test1(i64 %trip, i1 %cond) {
; CHECK-LABEL: test1
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[TRIP:%.*]], -1
; CHECK-NEXT:    [[XTRAITER:%.*]] = and i64 [[TRIP]], 7
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i64 [[TMP0]], 7
; CHECK-NEXT:    br i1 [[TMP1]], label %exit2.loopexit.unr-lcssa, label [[ENTRY_NEW:%.*]]
; CHECK:       entry.new:
; CHECK-NEXT:    [[UNROLL_ITER:%.*]] = sub i64 [[TRIP]], [[XTRAITER]]
; CHECK-NEXT:    br label [[LOOP_HEADER:%.*]]
; CHECK-LABEL:  loop_latch.epil:
; CHECK-NEXT:     %epil.iter.sub = add i64 %epil.iter, -1
; CHECK-NEXT:     %epil.iter.cmp = icmp eq i64 %epil.iter.sub, 0
; CHECK-NEXT:     br i1 %epil.iter.cmp, label %exit2.loopexit.epilog-lcssa, label %loop_header.epil
; CHECK-LABEL:  loop_latch.7:
; CHECK-NEXT:     %niter.nsub.7 = add i64 %niter, -8
; CHECK-NEXT:     %niter.ncmp.7 = icmp eq i64 %niter.nsub.7, 0
; CHECK-NEXT:     br i1 %niter.ncmp.7, label %exit2.loopexit.unr-lcssa.loopexit, label %loop_header
entry:
  br label %loop_header

loop_header:
  %iv = phi i64 [ 0, %entry ], [ %iv_next, %loop_latch ]
  br i1 %cond, label %loop_latch, label %loop_exiting_bb1

loop_exiting_bb1:
  br i1 false, label %loop_exiting_bb2, label %exit1

loop_exiting_bb2:
  br i1 false, label %loop_latch, label %exit3

exit3:
  ret void

loop_latch:
  %iv_next = add i64 %iv, 1
  %cmp = icmp ne i64 %iv_next, %trip
  br i1 %cmp, label %loop_header, label %exit2.loopexit

exit1:
 ret void

exit2.loopexit:
  ret void
}


; test with three exiting and two exit blocks.
; The non-latch exit block has 2 unique predecessors.
; There are 2 values passed to the exit blocks that are calculated at every iteration.
; %sum.02 and %add. Both of these are incoming values for phi from every exiting
; unrolled block.
define i32 @test2(i32* nocapture %a, i64 %n) {
; CHECK-LABEL: test2
; CHECK-LABEL: for.exit2.loopexit:
; CHECK-NEXT:    %retval.ph = phi i32 [ 42, %for.exiting_block ], [ %sum.02, %header ], [ %add, %for.body ], [ 42, %for.exiting_block.1 ], [ %add.1, %for.body.1 ], [ 42, %for.exiting_block.2 ], [ %add.2, %for.body.2 ], [ 42, %for.exiting_block.3 ],
; CHECK-NEXT:    br label %for.exit2
; CHECK-LABEL: for.exit2.loopexit2:
; CHECK-NEXT:    %retval.ph3 = phi i32 [ 42, %for.exiting_block.epil ], [ %sum.02.epil, %header.epil ]
; CHECK-NEXT:    br label %for.exit2
; CHECK-LABEL: for.exit2:
; CHECK-NEXT:    %retval = phi i32 [ %retval.ph, %for.exit2.loopexit ], [ %retval.ph3, %for.exit2.loopexit2 ]
; CHECK-NEXT:    ret i32 %retval
; CHECK: %niter.nsub.7 = add i64 %niter, -8
entry:
  br label %header

header:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  br i1 false, label %for.exit2, label %for.exiting_block

for.exiting_block:
 %cmp = icmp eq i64 %n, 42
 br i1 %cmp, label %for.exit2, label %for.body

for.body:
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %for.end, label %header

for.end:                                          ; preds = %for.body
  %sum.0.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %sum.0.lcssa

for.exit2:
  %retval = phi i32 [ %sum.02, %header ], [ 42, %for.exiting_block ]
  ret i32 %retval
}

; test with two exiting and three exit blocks.
; the non-latch exiting block has a switch.
define void @test3(i64 %trip, i64 %add) {
; CHECK-LABEL: test3
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[TRIP:%.*]], -1
; CHECK-NEXT:    [[XTRAITER:%.*]] = and i64 [[TRIP]], 7
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i64 [[TMP0]], 7
; CHECK-NEXT:    br i1 [[TMP1]], label %exit2.loopexit.unr-lcssa, label [[ENTRY_NEW:%.*]]
; CHECK:       entry.new:
; CHECK-NEXT:    %unroll_iter = sub i64 [[TRIP]], [[XTRAITER]]
; CHECK-NEXT:    br label [[LOOP_HEADER:%.*]]
; CHECK-LABEL:  loop_header:
; CHECK-NEXT:     %sum = phi i64 [ 0, %entry.new ], [ %sum.next.7, %loop_latch.7 ]
; CHECK-NEXT:     %niter = phi i64 [ %unroll_iter, %entry.new ], [ %niter.nsub.7, %loop_latch.7 ]
; CHECK-LABEL:  loop_exiting_bb1.7:
; CHECK-NEXT:     switch i64 %sum.next.6, label %loop_latch.7
; CHECK-LABEL:  loop_latch.7:
; CHECK-NEXT:     %sum.next.7 = add i64 %sum.next.6, %add
; CHECK-NEXT:     %niter.nsub.7 = add i64 %niter, -8
; CHECK-NEXT:     %niter.ncmp.7 = icmp eq i64 %niter.nsub.7, 0
; CHECK-NEXT:     br i1 %niter.ncmp.7, label %exit2.loopexit.unr-lcssa.loopexit, label %loop_header
entry:
  br label %loop_header

loop_header:
  %iv = phi i64 [ 0, %entry ], [ %iv_next, %loop_latch ]
  %sum = phi i64 [ 0, %entry ], [ %sum.next, %loop_latch ]
  br i1 undef, label %loop_latch, label %loop_exiting_bb1

loop_exiting_bb1:
   switch i64 %sum, label %loop_latch [
     i64 24, label %exit1
     i64 42, label %exit3
   ]

exit3:
  ret void

loop_latch:
  %iv_next = add nuw nsw i64 %iv, 1
  %sum.next = add i64 %sum, %add
  %cmp = icmp ne i64 %iv_next, %trip
  br i1 %cmp, label %loop_header, label %exit2.loopexit

exit1:
 ret void

exit2.loopexit:
  ret void
}

; FIXME: Support multiple exiting blocks to the same latch exit block.
define i32 @test4(i32* nocapture %a, i64 %n, i1 %cond) {
; CHECK-LABEL: test4
; CHECK-NOT: .unr
; CHECK-NOT: .epil
entry:
  br label %header

header:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  br i1 %cond, label %for.end, label %for.exiting_block

for.exiting_block:
 %cmp = icmp eq i64 %n, 42
 br i1 %cmp, label %for.exit2, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %for.end, label %header

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %header ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa

for.exit2:
  ret i32 42
}

; two exiting and two exit blocks.
; the non-latch exiting block has duplicate edges to the non-latch exit block.
define i64 @test5(i64 %trip, i64 %add, i1 %cond) {
; CHECK-LABEL: test5
; CHECK-LABEL:   exit1.loopexit:
; CHECK-NEXT:      %result.ph = phi i64 [ %ivy, %loop_exiting ], [ %ivy, %loop_exiting ], [ %ivy.1, %loop_exiting.1 ], [ %ivy.1, %loop_exiting.1 ], [ %ivy.2, %loop_exiting.2 ],
; CHECK-NEXT:      br label %exit1
; CHECK-LABEL:   exit1.loopexit2:
; CHECK-NEXT:      %ivy.epil = add i64 %iv.epil, %add
; CHECK-NEXT:      br label %exit1
; CHECK-LABEL:   exit1:
; CHECK-NEXT:      %result = phi i64 [ %result.ph, %exit1.loopexit ], [ %ivy.epil, %exit1.loopexit2 ]
; CHECK-NEXT:      ret i64 %result
; CHECK-LABEL:   loop_latch.7:
; CHECK: %niter.nsub.7 = add i64 %niter, -8
entry:
  br label %loop_header

loop_header:
  %iv = phi i64 [ 0, %entry ], [ %iv_next, %loop_latch ]
  %sum = phi i64 [ 0, %entry ], [ %sum.next, %loop_latch ]
  br i1 %cond, label %loop_latch, label %loop_exiting

loop_exiting:
   %ivy = add i64 %iv, %add
   switch i64 %sum, label %loop_latch [
     i64 24, label %exit1
     i64 42, label %exit1
   ]

loop_latch:
  %iv_next = add nuw nsw i64 %iv, 1
  %sum.next = add i64 %sum, %add
  %cmp = icmp ne i64 %iv_next, %trip
  br i1 %cmp, label %loop_header, label %latchexit

exit1:
 %result = phi i64 [ %ivy, %loop_exiting ], [ %ivy, %loop_exiting ]
 ret i64 %result

latchexit:
  ret i64 %sum.next
}

; test when exit blocks have successors.
define i32 @test6(i32* nocapture %a, i64 %n, i1 %cond, i32 %x) {
; CHECK-LABEL: test6
; CHECK-LABEL:   for.exit2.loopexit:
; CHECK-NEXT:      %retval.ph = phi i32 [ 42, %for.exiting_block ], [ %sum.02, %header ], [ %add, %latch ], [ 42, %for.exiting_block.1 ], [ %add.1, %latch.1 ], [ 42, %for.exiting_block.2 ], [ %add.2, %latch.2 ],
; CHECK-NEXT:      br label %for.exit2
; CHECK-LABEL:   for.exit2.loopexit2:
; CHECK-NEXT:      %retval.ph3 = phi i32 [ 42, %for.exiting_block.epil ], [ %sum.02.epil, %header.epil ]
; CHECK-NEXT:      br label %for.exit2
; CHECK-LABEL:   for.exit2:
; CHECK-NEXT:      %retval = phi i32 [ %retval.ph, %for.exit2.loopexit ], [ %retval.ph3, %for.exit2.loopexit2 ]
; CHECK-NEXT:      br i1 %cond, label %exit_true, label %exit_false
; CHECK-LABEL:   latch.7:
; CHECK:           %niter.nsub.7 = add i64 %niter, -8
entry:
  br label %header

header:
  %indvars.iv = phi i64 [ %indvars.iv.next, %latch ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %latch ], [ 0, %entry ]
  br i1 false, label %for.exit2, label %for.exiting_block

for.exiting_block:
 %cmp = icmp eq i64 %n, 42
 br i1 %cmp, label %for.exit2, label %latch

latch:
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %load = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %load, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %latch_exit, label %header

latch_exit:
  %sum.0.lcssa = phi i32 [ %add, %latch ]
  ret i32 %sum.0.lcssa

for.exit2:
  %retval = phi i32 [ %sum.02, %header ], [ 42, %for.exiting_block ]
  %addx = add i32 %retval, %x
  br i1 %cond, label %exit_true, label %exit_false

exit_true:
  ret i32 %retval

exit_false:
  ret i32 %addx
}
