; RUN: opt < %s -loop-unroll -unroll-runtime=true -unroll-runtime-epilog=true -unroll-runtime-multi-exit=true -verify-dom-info -verify-loop-info -S | FileCheck %s -check-prefix=EPILOG-NO-IC
; RUN: opt < %s -loop-unroll -unroll-runtime=true -unroll-runtime-epilog=true -unroll-runtime-multi-exit=true -verify-dom-info -verify-loop-info -instcombine -S | FileCheck %s -check-prefix=EPILOG
; RUN: opt < %s -loop-unroll -unroll-runtime -unroll-count=2 -unroll-runtime-epilog=true -unroll-runtime-multi-exit=true -verify-dom-info -verify-loop-info -instcombine
; RUN: opt < %s -loop-unroll -unroll-runtime=true -unroll-runtime-epilog=false -unroll-runtime-multi-exit=true -verify-dom-info -verify-loop-info -instcombine -S | FileCheck %s -check-prefix=PROLOG
; RUN: opt < %s -loop-unroll -unroll-runtime -unroll-runtime-epilog=false -unroll-count=2 -unroll-runtime-multi-exit=true -verify-dom-info -verify-loop-info -instcombine

; the third and fifth RUNs generate an epilog/prolog remainder block for all the test
; cases below (it does not generate a loop).

; test with three exiting and three exit blocks.
; none of the exit blocks have successors
define void @test1(i64 %trip, i1 %cond) {
; EPILOG: test1(
; EPILOG-NEXT:  entry:
; EPILOG-NEXT:    [[TMP0:%.*]] = add i64 [[TRIP:%.*]], -1
; EPILOG-NEXT:    [[XTRAITER:%.*]] = and i64 [[TRIP]], 7
; EPILOG-NEXT:    [[TMP1:%.*]] = icmp ult i64 [[TMP0]], 7
; EPILOG-NEXT:    br i1 [[TMP1]], label %exit2.loopexit.unr-lcssa, label [[ENTRY_NEW:%.*]]
; EPILOG:       entry.new:
; EPILOG-NEXT:    [[UNROLL_ITER:%.*]] = sub i64 [[TRIP]], [[XTRAITER]]
; EPILOG-NEXT:    br label [[LOOP_HEADER:%.*]]
; EPILOG:  loop_latch.epil:
; EPILOG-NEXT:     %epil.iter.sub = add i64 %epil.iter, -1
; EPILOG-NEXT:     %epil.iter.cmp = icmp eq i64 %epil.iter.sub, 0
; EPILOG-NEXT:     br i1 %epil.iter.cmp, label %exit2.loopexit.epilog-lcssa, label %loop_header.epil
; EPILOG:  loop_latch.7:
; EPILOG-NEXT:     %niter.nsub.7 = add i64 %niter, -8
; EPILOG-NEXT:     %niter.ncmp.7 = icmp eq i64 %niter.nsub.7, 0
; EPILOG-NEXT:     br i1 %niter.ncmp.7, label %exit2.loopexit.unr-lcssa.loopexit, label %loop_header

; PROLOG: test1(
; PROLOG-NEXT:  entry:
; PROLOG-NEXT:    [[TMP0:%.*]] = add i64 [[TRIP:%.*]], -1
; PROLOG-NEXT:    [[XTRAITER:%.*]] = and i64 [[TRIP]], 7
; PROLOG-NEXT:    [[TMP1:%.*]] = icmp eq i64 [[XTRAITER]], 0
; PROLOG-NEXT:    br i1 [[TMP1]], label %loop_header.prol.loopexit, label %loop_header.prol.preheader
; PROLOG:       loop_header.prol:
; PROLOG-NEXT:    %iv.prol = phi i64 [ 0, %loop_header.prol.preheader ], [ %iv_next.prol, %loop_latch.prol ]
; PROLOG-NEXT:    %prol.iter = phi i64 [ [[XTRAITER]], %loop_header.prol.preheader ], [ %prol.iter.sub, %loop_latch.prol ]
; PROLOG-NEXT:    br i1 %cond, label %loop_latch.prol, label %loop_exiting_bb1.prol
; PROLOG:       loop_latch.prol:
; PROLOG-NEXT:    %iv_next.prol = add i64 %iv.prol, 1
; PROLOG-NEXT:    %prol.iter.sub = add i64 %prol.iter, -1
; PROLOG-NEXT:    %prol.iter.cmp = icmp eq i64 %prol.iter.sub, 0
; PROLOG-NEXT:    br i1 %prol.iter.cmp, label %loop_header.prol.loopexit.unr-lcssa, label %loop_header.prol
; PROLOG:  loop_latch.7:
; PROLOG-NEXT:     %iv_next.7 = add i64 %iv, 8
; PROLOG-NEXT:     %cmp.7 = icmp eq i64 %iv_next.7, %trip
; PROLOG-NEXT:     br i1 %cmp.7, label %exit2.loopexit.unr-lcssa, label %loop_header
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
; EPILOG: test2(
; EPILOG: for.exit2.loopexit:
; EPILOG-NEXT:    %retval.ph = phi i32 [ 42, %for.exiting_block ], [ %sum.02, %header ], [ %add, %for.body ], [ 42, %for.exiting_block.1 ], [ %add.1, %for.body.1 ], [ 42, %for.exiting_block.2 ], [ %add.2, %for.body.2 ], [ 42, %for.exiting_block.3 ],
; EPILOG-NEXT:    br label %for.exit2
; EPILOG: for.exit2.loopexit2:
; EPILOG-NEXT:    %retval.ph3 = phi i32 [ 42, %for.exiting_block.epil ], [ %sum.02.epil, %header.epil ]
; EPILOG-NEXT:    br label %for.exit2
; EPILOG: for.exit2:
; EPILOG-NEXT:    %retval = phi i32 [ %retval.ph, %for.exit2.loopexit ], [ %retval.ph3, %for.exit2.loopexit2 ]
; EPILOG-NEXT:    ret i32 %retval
; EPILOG: %niter.nsub.7 = add i64 %niter, -8

; PROLOG: test2(
; PROLOG: for.exit2.loopexit:
; PROLOG-NEXT:    %retval.ph = phi i32 [ 42, %for.exiting_block ], [ %sum.02, %header ], [ %add, %for.body ], [ 42, %for.exiting_block.1 ], [ %add.1, %for.body.1 ], [ 42, %for.exiting_block.2 ], [ %add.2, %for.body.2 ], [ 42, %for.exiting_block.3 ],
; PROLOG-NEXT:    br label %for.exit2
; PROLOG: for.exit2.loopexit1:
; PROLOG-NEXT:    %retval.ph2 = phi i32 [ 42, %for.exiting_block.prol ], [ %sum.02.prol, %header.prol ]
; PROLOG-NEXT:    br label %for.exit2
; PROLOG: for.exit2:
; PROLOG-NEXT:    %retval = phi i32 [ %retval.ph, %for.exit2.loopexit ], [ %retval.ph2, %for.exit2.loopexit1 ]
; PROLOG-NEXT:    ret i32 %retval
; PROLOG: %indvars.iv.next.7 = add i64 %indvars.iv, 8

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
; EPILOG: test3(
; EPILOG-NEXT:  entry:
; EPILOG-NEXT:    [[TMP0:%.*]] = add i64 [[TRIP:%.*]], -1
; EPILOG-NEXT:    [[XTRAITER:%.*]] = and i64 [[TRIP]], 7
; EPILOG-NEXT:    [[TMP1:%.*]] = icmp ult i64 [[TMP0]], 7
; EPILOG-NEXT:    br i1 [[TMP1]], label %exit2.loopexit.unr-lcssa, label [[ENTRY_NEW:%.*]]
; EPILOG:       entry.new:
; EPILOG-NEXT:    %unroll_iter = sub i64 [[TRIP]], [[XTRAITER]]
; EPILOG-NEXT:    br label [[LOOP_HEADER:%.*]]
; EPILOG:  loop_header:
; EPILOG-NEXT:     %sum = phi i64 [ 0, %entry.new ], [ %sum.next.7, %loop_latch.7 ]
; EPILOG-NEXT:     %niter = phi i64 [ %unroll_iter, %entry.new ], [ %niter.nsub.7, %loop_latch.7 ]
; EPILOG:  loop_exiting_bb1.7:
; EPILOG-NEXT:     switch i64 %sum.next.6, label %loop_latch.7
; EPILOG:  loop_latch.7:
; EPILOG-NEXT:     %sum.next.7 = add i64 %sum.next.6, %add
; EPILOG-NEXT:     %niter.nsub.7 = add i64 %niter, -8
; EPILOG-NEXT:     %niter.ncmp.7 = icmp eq i64 %niter.nsub.7, 0
; EPILOG-NEXT:     br i1 %niter.ncmp.7, label %exit2.loopexit.unr-lcssa.loopexit, label %loop_header

; PROLOG:  test3(
; PROLOG-NEXT:  entry:
; PROLOG-NEXT:    [[TMP0:%.*]] = add i64 [[TRIP:%.*]], -1
; PROLOG-NEXT:    [[XTRAITER:%.*]] = and i64 [[TRIP]], 7
; PROLOG-NEXT:    [[TMP1:%.*]] = icmp eq i64 [[XTRAITER]], 0
; PROLOG-NEXT:    br i1 [[TMP1]], label %loop_header.prol.loopexit, label %loop_header.prol.preheader
; PROLOG:  loop_header:
; PROLOG-NEXT:     %iv = phi i64 [ %iv.unr, %entry.new ], [ %iv_next.7, %loop_latch.7 ]
; PROLOG-NEXT:     %sum = phi i64 [ %sum.unr, %entry.new ], [ %sum.next.7, %loop_latch.7 ]
; PROLOG:  loop_exiting_bb1.7:
; PROLOG-NEXT:     switch i64 %sum.next.6, label %loop_latch.7
; PROLOG:  loop_latch.7:
; PROLOG-NEXT:     %iv_next.7 = add nsw i64 %iv, 8
; PROLOG-NEXT:     %sum.next.7 = add i64 %sum.next.6, %add
; PROLOG-NEXT:     %cmp.7 = icmp eq i64 %iv_next.7, %trip
; PROLOG-NEXT:     br i1 %cmp.7, label %exit2.loopexit.unr-lcssa, label %loop_header
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
; EPILOG: test4(
; EPILOG-NOT: .unr
; EPILOG-NOT: .epil

; PROLOG: test4(
; PROLOG-NOT: .unr
; PROLOG-NOT: .prol
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

; FIXME: Support multiple exiting blocks to the unique exit block.
define void @unique_exit(i32 %arg) {
; EPILOG: unique_exit(
; EPILOG-NOT: .unr
; EPILOG-NOT: .epil

; PROLOG: unique_exit(
; PROLOG-NOT: .unr
; PROLOG-NOT: .prol
entry:
  %tmp = icmp sgt i32 undef, %arg
  br i1 %tmp, label %preheader, label %returnblock

preheader:                                 ; preds = %entry
  br label %header

LoopExit:                                ; preds = %header, %latch
  %tmp2.ph = phi i32 [ %tmp4, %header ], [ -1, %latch ]
  br label %returnblock

returnblock:                                         ; preds = %LoopExit, %entry
  %tmp2 = phi i32 [ -1, %entry ], [ %tmp2.ph, %LoopExit ]
  ret void

header:                                           ; preds = %preheader, %latch
  %tmp4 = phi i32 [ %inc, %latch ], [ %arg, %preheader ]
  %inc = add nsw i32 %tmp4, 1
  br i1 true, label %LoopExit, label %latch

latch:                                            ; preds = %header
  %cmp = icmp slt i32 %inc, undef
  br i1 %cmp, label %header, label %LoopExit
}

; two exiting and two exit blocks.
; the non-latch exiting block has duplicate edges to the non-latch exit block.
define i64 @test5(i64 %trip, i64 %add, i1 %cond) {
; EPILOG: test5(
; EPILOG:   exit1.loopexit:
; EPILOG-NEXT:      %result.ph = phi i64 [ %ivy, %loop_exiting ], [ %ivy, %loop_exiting ], [ %ivy.1, %loop_exiting.1 ], [ %ivy.1, %loop_exiting.1 ], [ %ivy.2, %loop_exiting.2 ],
; EPILOG-NEXT:      br label %exit1
; EPILOG:   exit1.loopexit2:
; EPILOG-NEXT:      %ivy.epil = add i64 %iv.epil, %add
; EPILOG-NEXT:      br label %exit1
; EPILOG:   exit1:
; EPILOG-NEXT:      %result = phi i64 [ %result.ph, %exit1.loopexit ], [ %ivy.epil, %exit1.loopexit2 ]
; EPILOG-NEXT:      ret i64 %result
; EPILOG:   loop_latch.7:
; EPILOG:      %niter.nsub.7 = add i64 %niter, -8

; PROLOG: test5(
; PROLOG:   exit1.loopexit:
; PROLOG-NEXT:      %result.ph = phi i64 [ %ivy, %loop_exiting ], [ %ivy, %loop_exiting ], [ %ivy.1, %loop_exiting.1 ], [ %ivy.1, %loop_exiting.1 ], [ %ivy.2, %loop_exiting.2 ],
; PROLOG-NEXT:      br label %exit1
; PROLOG:   exit1.loopexit1:
; PROLOG-NEXT:      %ivy.prol = add i64 %iv.prol, %add
; PROLOG-NEXT:      br label %exit1
; PROLOG:   exit1:
; PROLOG-NEXT:      %result = phi i64 [ %result.ph, %exit1.loopexit ], [ %ivy.prol, %exit1.loopexit1 ]
; PROLOG-NEXT:      ret i64 %result
; PROLOG:   loop_latch.7:
; PROLOG:      %iv_next.7 = add nsw i64 %iv, 8
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
; EPILOG: test6(
; EPILOG:   for.exit2.loopexit:
; EPILOG-NEXT:      %retval.ph = phi i32 [ 42, %for.exiting_block ], [ %sum.02, %header ], [ %add, %latch ], [ 42, %for.exiting_block.1 ], [ %add.1, %latch.1 ], [ 42, %for.exiting_block.2 ], [ %add.2, %latch.2 ],
; EPILOG-NEXT:      br label %for.exit2
; EPILOG:   for.exit2.loopexit2:
; EPILOG-NEXT:      %retval.ph3 = phi i32 [ 42, %for.exiting_block.epil ], [ %sum.02.epil, %header.epil ]
; EPILOG-NEXT:      br label %for.exit2
; EPILOG:   for.exit2:
; EPILOG-NEXT:      %retval = phi i32 [ %retval.ph, %for.exit2.loopexit ], [ %retval.ph3, %for.exit2.loopexit2 ]
; EPILOG-NEXT:      br i1 %cond, label %exit_true, label %exit_false
; EPILOG:   latch.7:
; EPILOG:           %niter.nsub.7 = add i64 %niter, -8

; PROLOG: test6(
; PROLOG:   for.exit2.loopexit:
; PROLOG-NEXT:      %retval.ph = phi i32 [ 42, %for.exiting_block ], [ %sum.02, %header ], [ %add, %latch ], [ 42, %for.exiting_block.1 ], [ %add.1, %latch.1 ], [ 42, %for.exiting_block.2 ], [ %add.2, %latch.2 ],
; PROLOG-NEXT:      br label %for.exit2
; PROLOG:   for.exit2.loopexit1:
; PROLOG-NEXT:      %retval.ph2 = phi i32 [ 42, %for.exiting_block.prol ], [ %sum.02.prol, %header.prol ]
; PROLOG-NEXT:      br label %for.exit2
; PROLOG:   for.exit2:
; PROLOG-NEXT:      %retval = phi i32 [ %retval.ph, %for.exit2.loopexit ], [ %retval.ph2, %for.exit2.loopexit1 ]
; PROLOG-NEXT:      br i1 %cond, label %exit_true, label %exit_false
; PROLOG: latch.7:
; PROLOG:   %indvars.iv.next.7 = add i64 %indvars.iv, 8
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

; test when value in exit block does not have VMap.
define i32 @test7(i32 %arg, i32 %arg1, i32 %arg2) {
; EPILOG-NO-IC: test7(
; EPILOG-NO-IC: loopexit1.loopexit:
; EPILOG-NO-IC-NEXT:  %sext3.ph = phi i32 [ %shft, %header ], [ %shft, %latch ], [ %shft, %latch.1 ], [ %shft, %latch.2 ], [ %shft, %latch.3 ], [ %shft, %latch.4 ], [ %shft, %latch.5 ], [ %shft, %latch.6 ]
; EPILOG-NO-IC-NEXT:  br label %loopexit1
; EPILOG-NO-IC: loopexit1.loopexit1:
; EPILOG-NO-IC-NEXT:  %sext3.ph2 = phi i32 [ %shft, %header.epil ]
; EPILOG-NO-IC-NEXT:  br label %loopexit1
; EPILOG-NO-IC: loopexit1:
; EPILOG-NO-IC-NEXT:   %sext3 = phi i32 [ %sext3.ph, %loopexit1.loopexit ], [ %sext3.ph2, %loopexit1.loopexit1 ]
bb:
  %tmp = icmp slt i32 undef, 2
  %sext = sext i32 undef to i64
  %shft = ashr exact i32 %arg, 16
  br i1 %tmp, label %loopexit2, label %preheader

preheader:                                              ; preds = %bb2
  br label %header

header:                                              ; preds = %latch, %preheader
  %tmp6 = phi i64 [ 1, %preheader ], [ %add, %latch ]
  br i1 false, label %loopexit1, label %latch

latch:                                              ; preds = %header
  %add = add nuw nsw i64 %tmp6, 1
  %tmp9 = icmp slt i64 %add, %sext
  br i1 %tmp9, label %header, label %latchexit

latchexit:                                             ; preds = %latch
  unreachable

loopexit2:                                             ; preds = %bb2
 ret i32 %shft

loopexit1:                                             ; preds = %header
  %sext3 = phi i32 [ %shft, %header ]
  ret i32 %sext3
}

; Nested loop and inner loop is unrolled
; FIXME: we cannot unroll with epilog remainder currently, because 
; the outer loop does not contain the epilog preheader and epilog exit (while
; infact it should). This causes us to choke up on LCSSA form being incorrect in
; outer loop. However, the exit block where LCSSA fails, is infact still within
; the outer loop. For now, we just bail out in presence of outer loop and epilog
; loop is generated.
; The outer loop header is the preheader for the inner loop and the inner header
; branches back to the outer loop.
define void @test8() {
; EPILOG: test8(
; EPILOG-NOT: niter

; PROLOG: test8(
; PROLOG: outerloop:
; PROLOG-NEXT: phi i64 [ 3, %bb ], [ 0, %outerloop.loopexit ]
; PROLOG:      %lcmp.mod = icmp eq i64
; PROLOG-NEXT: br i1 %lcmp.mod, label %innerH.prol.loopexit, label %innerH.prol.preheader
; PROLOG: latch.6:
; PROLOG-NEXT: %tmp4.7 = add nsw i64 %tmp3, 8
; PROLOG-NEXT: br i1 false, label %outerloop.loopexit.loopexit, label %latch.7
; PROLOG: latch.7
; PROLOG-NEXT: %tmp6.7 = icmp ult i64 %tmp4.7, 100
; PROLOG-NEXT: br i1 %tmp6.7, label %innerH, label %exit.unr-lcssa
bb:
  br label %outerloop

outerloop:                                              ; preds = %innerH, %bb
  %tmp = phi i64 [ 3, %bb ], [ 0, %innerH ]
  br label %innerH

innerH:                                              ; preds = %latch, %outerloop
  %tmp3 = phi i64 [ %tmp4, %latch ], [ %tmp, %outerloop ]
  %tmp4 = add nuw nsw i64 %tmp3, 1
  br i1 false, label %outerloop, label %latch

latch:                                              ; preds = %innerH
  %tmp6 = icmp ult i64 %tmp4, 100
  br i1 %tmp6, label %innerH, label %exit

exit:                                              ; preds = %latch
  ret void
}
