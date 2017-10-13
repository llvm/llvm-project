; RUN: opt -S -loop-predication < %s 2>&1 | FileCheck %s
; RUN: opt -S -passes='require<scalar-evolution>,loop(loop-predication)' < %s 2>&1 | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

define i32 @signed_loop_0_to_n_nested_0_to_l_inner_index_check(i32* %array, i32 %length, i32 %n, i32 %l) {
; CHECK-LABEL: @signed_loop_0_to_n_nested_0_to_l_inner_index_check
entry:
  %tmp5 = icmp sle i32 %n, 0
  br i1 %tmp5, label %exit, label %outer.loop.preheader

outer.loop.preheader:
  br label %outer.loop

outer.loop:
  %outer.loop.acc = phi i32 [ %outer.loop.acc.next, %outer.loop.inc ], [ 0, %outer.loop.preheader ]
  %i = phi i32 [ %i.next, %outer.loop.inc ], [ 0, %outer.loop.preheader ]
  %tmp6 = icmp sle i32 %l, 0
  br i1 %tmp6, label %outer.loop.inc, label %inner.loop.preheader
  
inner.loop.preheader:
; CHECK: inner.loop.preheader:
; CHECK: [[limit_check:[^ ]+]] = icmp sle i32 %l, %length
; CHECK-NEXT: [[first_iteration_check:[^ ]+]] = icmp ult i32 0, %length
; CHECK-NEXT: [[wide_cond:[^ ]+]] = and i1 [[first_iteration_check]], [[limit_check]]
; CHECK-NEXT: br label %inner.loop
  br label %inner.loop

inner.loop:
; CHECK: inner.loop:
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 [[wide_cond]], i32 9) [ "deopt"() ]
  %inner.loop.acc = phi i32 [ %inner.loop.acc.next, %inner.loop ], [ %outer.loop.acc, %inner.loop.preheader ]
  %j = phi i32 [ %j.next, %inner.loop ], [ 0, %inner.loop.preheader ]

  %within.bounds = icmp ult i32 %j, %length
  call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]
  
  %j.i64 = zext i32 %j to i64
  %array.j.ptr = getelementptr inbounds i32, i32* %array, i64 %j.i64
  %array.j = load i32, i32* %array.j.ptr, align 4
  %inner.loop.acc.next = add i32 %inner.loop.acc, %array.j

  %j.next = add nsw i32 %j, 1
  %inner.continue = icmp slt i32 %j.next, %l
  br i1 %inner.continue, label %inner.loop, label %outer.loop.inc

outer.loop.inc:
  %outer.loop.acc.next = phi i32 [ %inner.loop.acc.next, %inner.loop ], [ %outer.loop.acc, %outer.loop ]
  %i.next = add nsw i32 %i, 1
  %outer.continue = icmp slt i32 %i.next, %n
  br i1 %outer.continue, label %outer.loop, label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %outer.loop.acc.next, %outer.loop.inc ]
  ret i32 %result
}

define i32 @signed_loop_0_to_n_nested_0_to_l_outer_index_check(i32* %array, i32 %length, i32 %n, i32 %l) {
; CHECK-LABEL: @signed_loop_0_to_n_nested_0_to_l_outer_index_check
entry:
  %tmp5 = icmp sle i32 %n, 0
  br i1 %tmp5, label %exit, label %outer.loop.preheader

outer.loop.preheader:
; CHECK: outer.loop.preheader:
; CHECK: [[limit_check:[^ ]+]] = icmp sle i32 %n, %length
; CHECK-NEXT: [[first_iteration_check:[^ ]+]] = icmp ult i32 0, %length
; CHECK-NEXT: [[wide_cond:[^ ]+]] = and i1 [[first_iteration_check]], [[limit_check]]
; CHECK-NEXT: br label %outer.loop
  br label %outer.loop

outer.loop:
  %outer.loop.acc = phi i32 [ %outer.loop.acc.next, %outer.loop.inc ], [ 0, %outer.loop.preheader ]
  %i = phi i32 [ %i.next, %outer.loop.inc ], [ 0, %outer.loop.preheader ]
  %tmp6 = icmp sle i32 %l, 0
  br i1 %tmp6, label %outer.loop.inc, label %inner.loop.preheader
  
inner.loop.preheader:
  br label %inner.loop

inner.loop:
; CHECK: inner.loop:
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 [[wide_cond]], i32 9) [ "deopt"() ]

  %inner.loop.acc = phi i32 [ %inner.loop.acc.next, %inner.loop ], [ %outer.loop.acc, %inner.loop.preheader ]
  %j = phi i32 [ %j.next, %inner.loop ], [ 0, %inner.loop.preheader ]

  %within.bounds = icmp ult i32 %i, %length
  call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]
  
  %i.i64 = zext i32 %i to i64
  %array.i.ptr = getelementptr inbounds i32, i32* %array, i64 %i.i64
  %array.i = load i32, i32* %array.i.ptr, align 4
  %inner.loop.acc.next = add i32 %inner.loop.acc, %array.i

  %j.next = add nsw i32 %j, 1
  %inner.continue = icmp slt i32 %j.next, %l
  br i1 %inner.continue, label %inner.loop, label %outer.loop.inc

outer.loop.inc:
  %outer.loop.acc.next = phi i32 [ %inner.loop.acc.next, %inner.loop ], [ %outer.loop.acc, %outer.loop ]
  %i.next = add nsw i32 %i, 1
  %outer.continue = icmp slt i32 %i.next, %n
  br i1 %outer.continue, label %outer.loop, label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %outer.loop.acc.next, %outer.loop.inc ]
  ret i32 %result
}

define i32 @signed_loop_0_to_n_nested_i_to_l_inner_index_check(i32* %array, i32 %length, i32 %n, i32 %l) {
; CHECK-LABEL: @signed_loop_0_to_n_nested_i_to_l_inner_index_check
entry:
  %tmp5 = icmp sle i32 %n, 0
  br i1 %tmp5, label %exit, label %outer.loop.preheader

outer.loop.preheader:
; CHECK: outer.loop.preheader:
; CHECK-NEXT: [[limit_check_outer:[^ ]+]] = icmp sle i32 %n, %length
; CHECK-NEXT: [[first_iteration_check_outer:[^ ]+]] = icmp ult i32 0, %length
; CHECK-NEXT: [[wide_cond_outer:[^ ]+]] = and i1 [[first_iteration_check_outer]], [[limit_check_outer]]
; CHECK-NEXT: br label %outer.loop
  br label %outer.loop

outer.loop:
; CHECK: outer.loop:
  %outer.loop.acc = phi i32 [ %outer.loop.acc.next, %outer.loop.inc ], [ 0, %outer.loop.preheader ]
  %i = phi i32 [ %i.next, %outer.loop.inc ], [ 0, %outer.loop.preheader ]
  %tmp6 = icmp sle i32 %l, 0
  br i1 %tmp6, label %outer.loop.inc, label %inner.loop.preheader
  
inner.loop.preheader:
; CHECK: inner.loop.preheader:
; CHECK: [[limit_check_inner:[^ ]+]] = icmp sle i32 %l, %length
; CHECK: br label %inner.loop
  br label %inner.loop

inner.loop:
; CHECK: inner.loop:
; CHECK: [[wide_cond:[^ ]+]] = and i1 [[limit_check_inner]], [[wide_cond_outer]]
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 [[wide_cond]], i32 9) [ "deopt"() ]
  %inner.loop.acc = phi i32 [ %inner.loop.acc.next, %inner.loop ], [ %outer.loop.acc, %inner.loop.preheader ]
  %j = phi i32 [ %j.next, %inner.loop ], [ %i, %inner.loop.preheader ]

  %within.bounds = icmp ult i32 %j, %length
  call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]
  
  %j.i64 = zext i32 %j to i64
  %array.j.ptr = getelementptr inbounds i32, i32* %array, i64 %j.i64
  %array.j = load i32, i32* %array.j.ptr, align 4
  %inner.loop.acc.next = add i32 %inner.loop.acc, %array.j

  %j.next = add nsw i32 %j, 1
  %inner.continue = icmp slt i32 %j.next, %l
  br i1 %inner.continue, label %inner.loop, label %outer.loop.inc

outer.loop.inc:
  %outer.loop.acc.next = phi i32 [ %inner.loop.acc.next, %inner.loop ], [ %outer.loop.acc, %outer.loop ]
  %i.next = add nsw i32 %i, 1
  %outer.continue = icmp slt i32 %i.next, %n
  br i1 %outer.continue, label %outer.loop, label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %outer.loop.acc.next, %outer.loop.inc ]
  ret i32 %result
}

define i32 @cant_expand_guard_check_start(i32* %array, i32 %length, i32 %n, i32 %l, i32 %maybezero) {
; CHECK-LABEL: @cant_expand_guard_check_start
entry:
  %tmp5 = icmp sle i32 %n, 0
  br i1 %tmp5, label %exit, label %outer.loop.preheader

outer.loop.preheader:
  br label %outer.loop

outer.loop:
  %outer.loop.acc = phi i32 [ %outer.loop.acc.next, %outer.loop.inc ], [ 0, %outer.loop.preheader ]
  %i = phi i32 [ %i.next, %outer.loop.inc ], [ 0, %outer.loop.preheader ]
  %tmp6 = icmp sle i32 %l, 0
  %div = udiv i32 %i, %maybezero
  br i1 %tmp6, label %outer.loop.inc, label %inner.loop.preheader
  
inner.loop.preheader:
; CHECK: inner.loop.preheader:
; CHECK: br label %inner.loop
  br label %inner.loop

inner.loop:
; CHECK: inner.loop:
; CHECK: %within.bounds = icmp ult i32 %j, %length
; CHECK: call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]
  %inner.loop.acc = phi i32 [ %inner.loop.acc.next, %inner.loop ], [ %outer.loop.acc, %inner.loop.preheader ]
  %j = phi i32 [ %j.next, %inner.loop ], [ %div, %inner.loop.preheader ]

  %within.bounds = icmp ult i32 %j, %length
  call void (i1, ...) @llvm.experimental.guard(i1 %within.bounds, i32 9) [ "deopt"() ]
  
  %j.i64 = zext i32 %j to i64
  %array.j.ptr = getelementptr inbounds i32, i32* %array, i64 %j.i64
  %array.j = load i32, i32* %array.j.ptr, align 4
  %inner.loop.acc.next = add i32 %inner.loop.acc, %array.j

  %j.next = add nsw i32 %j, 1
  %inner.continue = icmp slt i32 %j.next, %l
  br i1 %inner.continue, label %inner.loop, label %outer.loop.inc

outer.loop.inc:
  %outer.loop.acc.next = phi i32 [ %inner.loop.acc.next, %inner.loop ], [ %outer.loop.acc, %outer.loop ]
  %i.next = add nsw i32 %i, 1
  %outer.continue = icmp slt i32 %i.next, %n
  br i1 %outer.continue, label %outer.loop, label %exit

exit:
  %result = phi i32 [ 0, %entry ], [ %outer.loop.acc.next, %outer.loop.inc ]
  ret i32 %result
}