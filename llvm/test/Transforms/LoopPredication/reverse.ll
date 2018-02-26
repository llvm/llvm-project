; RUN: opt -S -loop-predication -loop-predication-enable-count-down-loop=true < %s 2>&1 | FileCheck %s
; RUN: opt -S -passes='require<scalar-evolution>,loop(loop-predication)' -loop-predication-enable-count-down-loop=true < %s 2>&1 | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

define i32 @signed_reverse_loop_n_to_lower_limit(i32* %array, i32 %length, i32 %n, i32 %lowerlimit) {
; CHECK-LABEL: @signed_reverse_loop_n_to_lower_limit(
entry:
  %tmp5 = icmp eq i32 %n, 0
  br i1 %tmp5, label %exit, label %loop.preheader

; CHECK:       loop.preheader:
; CHECK-NEXT:    [[range_start:%.*]] = add i32 %n, -1
; CHECK-NEXT:    [[first_iteration_check:%.*]] = icmp ult i32 [[range_start]], %length
; CHECK-NEXT:    [[no_wrap_check:%.*]] = icmp sge i32 %lowerlimit, 1
; CHECK-NEXT:    [[wide_cond:%.*]] = and i1 [[first_iteration_check]], [[no_wrap_check]]
loop.preheader:
  br label %loop

; CHECK: loop:
; CHECK:    call void (i1, ...) @llvm.experimental.guard(i1 [[wide_cond]], i32 9) [ "deopt"() ]
loop:
  %loop.acc = phi i32 [ %loop.acc.next, %loop ], [ 0, %loop.preheader ]
  %i = phi i32 [ %i.next, %loop ], [ %n, %loop.preheader ]
  %i.next = add nsw i32 %i, -1
  %within.bounds = icmp ult i32 %i.next, %length
  call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]
  %i.i64 = zext i32 %i.next to i64
  %array.i.ptr = getelementptr inbounds i32, i32* %array, i64 %i.i64
  %array.i = load i32, i32* %array.i.ptr, align 4
  %loop.acc.next = add i32 %loop.acc, %array.i
  %continue = icmp sgt i32 %i, %lowerlimit
  br i1 %continue, label %loop, label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %loop.acc.next, %loop ]
  ret i32 %result
}

define i32 @unsigned_reverse_loop_n_to_lower_limit(i32* %array, i32 %length, i32 %n, i32 %lowerlimit) {
; CHECK-LABEL: @unsigned_reverse_loop_n_to_lower_limit(
entry:
  %tmp5 = icmp eq i32 %n, 0
  br i1 %tmp5, label %exit, label %loop.preheader

; CHECK:       loop.preheader:
; CHECK-NEXT:    [[range_start:%.*]] = add i32 %n, -1
; CHECK-NEXT:    [[first_iteration_check:%.*]] = icmp ult i32 [[range_start]], %length
; CHECK-NEXT:    [[no_wrap_check:%.*]] = icmp uge i32 %lowerlimit, 1
; CHECK-NEXT:    [[wide_cond:%.*]] = and i1 [[first_iteration_check]], [[no_wrap_check]]
loop.preheader:
  br label %loop

; CHECK: loop:
; CHECK:    call void (i1, ...) @llvm.experimental.guard(i1 [[wide_cond]], i32 9) [ "deopt"() ]
loop:
  %loop.acc = phi i32 [ %loop.acc.next, %loop ], [ 0, %loop.preheader ]
  %i = phi i32 [ %i.next, %loop ], [ %n, %loop.preheader ]
  %i.next = add nsw i32 %i, -1
  %within.bounds = icmp ult i32 %i.next, %length
  call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]
  %i.i64 = zext i32 %i.next to i64
  %array.i.ptr = getelementptr inbounds i32, i32* %array, i64 %i.i64
  %array.i = load i32, i32* %array.i.ptr, align 4
  %loop.acc.next = add i32 %loop.acc, %array.i
  %continue = icmp ugt i32 %i, %lowerlimit
  br i1 %continue, label %loop, label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %loop.acc.next, %loop ]
  ret i32 %result
}


; if we predicated the loop, the guard will definitely fail and we will
; deoptimize early on.
define i32 @unsigned_reverse_loop_n_to_0(i32* %array, i32 %length, i32 %n, i32 %lowerlimit) {
; CHECK-LABEL: @unsigned_reverse_loop_n_to_0(
entry:
  %tmp5 = icmp eq i32 %n, 0
  br i1 %tmp5, label %exit, label %loop.preheader

; CHECK:       loop.preheader:
; CHECK-NEXT:    [[range_start:%.*]] = add i32 %n, -1
; CHECK-NEXT:    [[first_iteration_check:%.*]] = icmp ult i32 [[range_start]], %length
; CHECK-NEXT:    [[wide_cond:%.*]] = and i1 [[first_iteration_check]], false
loop.preheader:
  br label %loop

; CHECK: loop:
; CHECK:    call void (i1, ...) @llvm.experimental.guard(i1 [[wide_cond]], i32 9) [ "deopt"() ]
loop:
  %loop.acc = phi i32 [ %loop.acc.next, %loop ], [ 0, %loop.preheader ]
  %i = phi i32 [ %i.next, %loop ], [ %n, %loop.preheader ]
  %i.next = add nsw i32 %i, -1
  %within.bounds = icmp ult i32 %i.next, %length
  call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]
  %i.i64 = zext i32 %i.next to i64
  %array.i.ptr = getelementptr inbounds i32, i32* %array, i64 %i.i64
  %array.i = load i32, i32* %array.i.ptr, align 4
  %loop.acc.next = add i32 %loop.acc, %array.i
  %continue = icmp ugt i32 %i, 0
  br i1 %continue, label %loop, label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %loop.acc.next, %loop ]
  ret i32 %result
}

; do not loop predicate when the range has step -1 and latch has step 1.
define i32 @reverse_loop_range_step_increment(i32 %n, i32* %array, i32 %length) {
; CHECK-LABEL: @reverse_loop_range_step_increment(
entry:
  %tmp5 = icmp eq i32 %n, 0
  br i1 %tmp5, label %exit, label %loop.preheader

loop.preheader:
  br label %loop

; CHECK: loop:
; CHECK: llvm.experimental.guard(i1 %within.bounds, i32 9)
loop:
  %loop.acc = phi i32 [ %loop.acc.next, %loop ], [ 0, %loop.preheader ]
  %i = phi i32 [ %i.next, %loop ], [ %n, %loop.preheader ]
  %irc = phi i32 [ %i.inc, %loop ], [ 1, %loop.preheader ]
  %i.inc = add nuw nsw i32 %irc, 1
  %within.bounds = icmp ult i32 %irc, %length
  call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]
  %i.i64 = zext i32 %irc to i64
  %array.i.ptr = getelementptr inbounds i32, i32* %array, i64 %i.i64
  %array.i = load i32, i32* %array.i.ptr, align 4
  %i.next = add nsw i32 %i, -1
  %loop.acc.next = add i32 %loop.acc, %array.i
  %continue = icmp ugt i32 %i, 65534
  br i1 %continue, label %loop, label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %loop.acc.next, %loop ]
  ret i32 %result
}
