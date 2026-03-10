; RUN: opt -S < %s -passes='loop-vectorize,verify<loops>,print<loops>' -disable-output 2>&1 | FileCheck %s

declare void @init_mem(ptr, i64);

; Tests that the additional middle.split created for handling loops with
; uncountable early exits is correctly adding to the outer loop at depth 1.
define void @early_exit_in_outer_loop1() {
; CHECK-LABEL: Loop info for function 'early_exit_in_outer_loop1':
; CHECK: Loop at depth 1 containing: %loop.outer<header>,%loop.inner.found,%loop.inner.end<latch>,%loop.inner.end.loopexit,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit
; CHECK:    Loop at depth 2 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  %p1 = alloca [1024 x i8]
  %p2 = alloca [1024 x i8]
  call void @init_mem(ptr %p1, i64 1024)
  call void @init_mem(ptr %p2, i64 1024)
  br label %loop.outer

loop.outer:
  %count = phi i64 [ 0, %entry ], [ %count.next, %loop.inner.end ]
  br label %loop.inner

loop.inner:
  %index = phi i64 [ %index.next, %loop.inner.inc ], [ 3, %loop.outer ]
  %arrayidx = getelementptr inbounds i8, ptr %p1, i64 %index
  %ld1 = load i8, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, ptr %p2, i64 %index
  %ld2 = load i8, ptr %arrayidx1, align 1
  %cmp3 = icmp eq i8 %ld1, %ld2
  br i1 %cmp3, label %loop.inner.inc, label %loop.inner.found

loop.inner.inc:
  %index.next = add i64 %index, 1
  %exitcond = icmp ne i64 %index.next, 67
  br i1 %exitcond, label %loop.inner, label %loop.inner.end

loop.inner.found:
  br label %loop.inner.end

loop.inner.end:
  %count.next = phi i64 [ 0, %loop.inner.inc ], [ 1, %loop.inner.found ]
  br label %loop.outer
}

; Tests that the additional middle.split created for handling loops with
; uncountable early exits is correctly adding to both the outer and middle
; loops at depths 1 and 2, respectively.
define void @early_exit_in_outer_loop2() {
; CHECK-LABEL: Loop info for function 'early_exit_in_outer_loop2':
; CHECK: Loop at depth 1 containing: %loop.outer<header>,%loop.middle,%loop.inner.found,%loop.inner.end,%loop.middle.end,%loop.outer.latch<latch>,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit
; CHECK:    Loop at depth 2 containing: %loop.middle<header>,%loop.inner.end<latch><exiting>,%vector.ph,%vector.body<exiting>,%vector.body.interim,%middle.block
; CHECK:        Loop at depth 3 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  %p1 = alloca [1024 x i8]
  %p2 = alloca [1024 x i8]
  call void @init_mem(ptr %p1, i64 1024)
  call void @init_mem(ptr %p2, i64 1024)
  br label %loop.outer

loop.outer:
  %count.outer = phi i64 [ 0, %entry ], [ %count.outer.next , %loop.outer.latch ]
  br label %loop.middle

loop.middle:
  br label %loop.inner

loop.inner:
  %index = phi i64 [ %index.next, %loop.inner.inc ], [ 3, %loop.middle ]
  %arrayidx = getelementptr inbounds i8, ptr %p1, i64 %index
  %ld1 = load i8, ptr %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, ptr %p2, i64 %index
  %ld2 = load i8, ptr %arrayidx1, align 1
  %cmp3 = icmp eq i8 %ld1, %ld2
  br i1 %cmp3, label %loop.inner.inc, label %loop.inner.found

loop.inner.inc:
  %index.next = add i64 %index, 1
  %exitcond = icmp ne i64 %index.next, 67
  br i1 %exitcond, label %loop.inner, label %loop.inner.end

loop.inner.end:
  br i1 false, label %loop.middle, label %loop.middle.end

loop.middle.end:
  br label %loop.outer.latch

loop.inner.found:
  br label %loop.outer.latch

loop.outer.latch:
  %t = phi i64 [ 0, %loop.middle.end ], [ 1, %loop.inner.found ]
  %count.outer.next = add i64 %count.outer, %t
  br label %loop.outer
}

define i32 @early_exit_branch_to_outer_header() {
; CHECK-LABEL: Loop info for function 'early_exit_branch_to_outer_header':
; CHECK-NEXT:  Loop at depth 1 containing: %outer.header<header>,%outer.header.loopexit<latch>,%vector.ph,%vector.body,%vector.body.interim<exiting>,%vector.early.exit
; CHECK-NEXT:    Loop at depth 2 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  %src = alloca [1024 x i8]
  call void @init_mem(ptr %src, i64 1024)
  br label %outer.header

outer.header:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %outer.header ], [ %iv.next, %loop.latch ]
  %gep.src = getelementptr i8, ptr %src, i64 %iv
  %l = load i8, ptr %gep.src, align 1
  %c = icmp eq i8 %l, 0
  br i1 %c, label %outer.header, label %loop.latch

loop.latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop.header

exit:
  ret i32 1
}

; Tests that when an early-exit inner loop has multiple exits and all exits
; leave the outer loop.
define i32 @multi_early_exit_all_leave_outer_loop(i1 %c) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_all_leave_outer_loop':
; CHECK: Loop at depth 1 containing: %outer.header<header>,%outer.latch<latch><exiting>,%vector.ph,%vector.body<exiting>,%vector.body.interim,%middle.block
; CHECK:    {{.*}}Loop at depth 2 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  br label %outer.header

outer.header:
  br label %inner.header

inner.header:
  %iv = phi i64 [ 0, %outer.header ], [ %iv.next, %inner.latch ]
  br i1 %c, label %early.exit.1, label %inner.body

inner.body:
  %trunc = trunc i64 %iv to i16
  %cmp = icmp ult i16 %trunc, 0
  br i1 %cmp, label %early.exit.2, label %inner.latch

inner.latch:
  %iv.next = add i64 %iv, 1
  %exit.cond = icmp ult i64 %iv, 1
  br i1 %exit.cond, label %inner.header, label %outer.latch

outer.latch:
  br i1 %c, label %early.exit.1, label %outer.header

early.exit.1:
  ret i32 1

early.exit.2:
  ret i32 0
}

; Tests that when an inner loop has two early exits at different loop levels
; (one staying in the outer loop, one leaving all loops).
define i32 @multi_early_exit_different_loop_levels(i1 %c) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_different_loop_levels':
; CHECK: Loop at depth 1 containing: %outer.header<header>,%early.exit.outer,%outer.latch<latch>,%outer.latch.loopexit,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.check<exiting>,%vector.early.exit.0
; CHECK:    {{.*}}Loop at depth 2 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  br label %outer.header

outer.header:
  br label %inner.header

inner.header:
  %iv = phi i64 [ 0, %outer.header ], [ %iv.next, %inner.latch ]
  br i1 %c, label %early.exit.outer, label %inner.body

inner.body:
  %trunc = trunc i64 %iv to i16
  %cmp = icmp ult i16 %trunc, 0
  br i1 %cmp, label %early.exit.leave, label %inner.latch

inner.latch:
  %iv.next = add i64 %iv, 1
  %exit.cond = icmp ult i64 %iv, 1
  br i1 %exit.cond, label %inner.header, label %outer.latch

outer.latch:
  br label %outer.header

early.exit.outer:
  br label %outer.latch

early.exit.leave:
  ret i32 0
}

; Same as above, but the early exit order is reversed: the first early exit
; leaves all loops and the second stays in the outer loop.
define i32 @multi_early_exit_different_loop_levels_reversed(i1 %c) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_different_loop_levels_reversed':
; CHECK: Loop at depth 1 containing: %outer.header<header>,%early.exit.outer,%outer.latch<latch>,%outer.latch.loopexit,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.check<exiting>,%vector.early.exit.1
; CHECK:    {{.*}}Loop at depth 2 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  br label %outer.header

outer.header:
  br label %inner.header

inner.header:
  %iv = phi i64 [ 0, %outer.header ], [ %iv.next, %inner.latch ]
  br i1 %c, label %early.exit.leave, label %inner.body

inner.body:
  %trunc = trunc i64 %iv to i16
  %cmp = icmp ult i16 %trunc, 0
  br i1 %cmp, label %early.exit.outer, label %inner.latch

inner.latch:
  %iv.next = add i64 %iv, 1
  %exit.cond = icmp ult i64 %iv, 1
  br i1 %exit.cond, label %inner.header, label %outer.latch

outer.latch:
  br label %outer.header

early.exit.outer:
  br label %outer.latch

early.exit.leave:
  ret i32 0
}

; Tests that when an inner loop has two early exits at different loop levels
; (one going to a middle loop, one going to the outer loop).
define i32 @multi_early_exit_two_different_loops(i1 %c) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_two_different_loops':
; CHECK: Loop at depth 1 containing: %outer.header<header>,%middle.header,%early.exit.outer,%early.exit.middle,%middle.latch,%outer.latch<latch>,%middle.latch.loopexit,%outer.latch.loopexit,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.check,%vector.early.exit.1,%vector.early.exit.0
; CHECK:    {{.*}}Loop at depth 2 containing: %middle.header<header>,%early.exit.middle,%middle.latch<latch><exiting>,%middle.latch.loopexit,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.check<exiting>,%vector.early.exit.0
; CHECK:        {{.*}}Loop at depth 3 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  br label %outer.header

outer.header:
  br label %middle.header

middle.header:
  br label %inner.header

inner.header:
  %iv = phi i64 [ 0, %middle.header ], [ %iv.next, %inner.latch ]
  br i1 %c, label %early.exit.middle, label %inner.body

inner.body:
  %trunc = trunc i64 %iv to i16
  %cmp = icmp ult i16 %trunc, 0
  br i1 %cmp, label %early.exit.outer, label %inner.latch

inner.latch:
  %iv.next = add i64 %iv, 1
  %exit.cond = icmp ult i64 %iv, 1
  br i1 %exit.cond, label %inner.header, label %middle.latch

middle.latch:
  br i1 %c, label %middle.header, label %outer.latch

outer.latch:
  br label %outer.header

early.exit.middle:
  br label %middle.latch

early.exit.outer:
  br label %outer.latch
}

; Same as above, but the early exit order is reversed: the first early exit
; goes to the outer loop and the second goes to the middle loop.
define i32 @multi_early_exit_two_different_loops_reversed(i1 %c) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_two_different_loops_reversed':
; CHECK: Loop at depth 1 containing: %outer.header<header>,%middle.header,%early.exit.middle,%middle.latch,%early.exit.outer,%outer.latch<latch>,%middle.latch.loopexit,%outer.latch.loopexit,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.check,%vector.early.exit.1,%vector.early.exit.0
; CHECK:    {{.*}}Loop at depth 2 containing: %middle.header<header>,%early.exit.middle,%middle.latch<latch><exiting>,%middle.latch.loopexit,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.check<exiting>,%vector.early.exit.1
; CHECK:        {{.*}}Loop at depth 3 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  br label %outer.header

outer.header:
  br label %middle.header

middle.header:
  br label %inner.header

inner.header:
  %iv = phi i64 [ 0, %middle.header ], [ %iv.next, %inner.latch ]
  br i1 %c, label %early.exit.outer, label %inner.body

inner.body:
  %trunc = trunc i64 %iv to i16
  %cmp = icmp ult i16 %trunc, 0
  br i1 %cmp, label %early.exit.middle, label %inner.latch

inner.latch:
  %iv.next = add i64 %iv, 1
  %exit.cond = icmp ult i64 %iv, 1
  br i1 %exit.cond, label %inner.header, label %middle.latch

middle.latch:
  br i1 %c, label %middle.header, label %outer.latch

outer.latch:
  br label %outer.header

early.exit.middle:
  br label %middle.latch

early.exit.outer:
  br label %outer.latch
}
