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

; Tests when an early-exit inner loop has multiple exits and all exits
; leave the outer loop.
define i32 @multi_early_exit_all_leave_outer_loop(i1 %c, ptr dereferenceable(1024) %src) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_all_leave_outer_loop':
; CHECK-NEXT: Loop at depth 1 containing: %outer.header<header>,%outer.latch<latch><exiting>,%vector.ph,%vector.body<exiting>,%vector.body.interim,%middle.block
; CHECK-NEXT:     Loop at depth 2 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  br label %outer.header

outer.header:
  br label %inner.header

inner.header:
  %iv = phi i64 [ 0, %outer.header ], [ %iv.next, %inner.latch ]
  br i1 %c, label %early.exit.1, label %inner.body

inner.body:
  %gep = getelementptr inbounds i8, ptr %src, i64 %iv
  %ld = load i8, ptr %gep, align 1
  %cmp = icmp eq i8 %ld, 0
  br i1 %cmp, label %early.exit.2, label %inner.latch

inner.latch:
  %iv.next = add i64 %iv, 1
  %exit.cond = icmp ult i64 %iv, 63
  br i1 %exit.cond, label %inner.header, label %outer.latch

outer.latch:
  br i1 %c, label %early.exit.1, label %outer.header

early.exit.1:
  ret i32 1

early.exit.2:
  ret i32 0
}

; Tests when an inner loop has two early exits at different loop levels
; (one staying in the outer loop, one leaving all loops).
define i32 @multi_early_exit_different_loop_levels(i1 %c, ptr dereferenceable(1024) %src) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_different_loop_levels':
; CHECK-NEXT: Loop at depth 1 containing: %outer.header<header>,%early.exit.outer,%outer.latch<latch>,%outer.latch.loopexit,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.check<exiting>,%vector.early.exit.0
; CHECK-NEXT:     Loop at depth 2 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  br label %outer.header

outer.header:
  br label %inner.header

inner.header:
  %iv = phi i64 [ 0, %outer.header ], [ %iv.next, %inner.latch ]
  br i1 %c, label %early.exit.outer, label %inner.body

inner.body:
  %gep = getelementptr inbounds i8, ptr %src, i64 %iv
  %ld = load i8, ptr %gep, align 1
  %cmp = icmp eq i8 %ld, 0
  br i1 %cmp, label %early.exit.leave, label %inner.latch

inner.latch:
  %iv.next = add i64 %iv, 1
  %exit.cond = icmp ult i64 %iv, 63
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
define i32 @multi_early_exit_different_loop_levels_reversed(i1 %c, ptr dereferenceable(1024) %src) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_different_loop_levels_reversed':
; CHECK-NEXT: Loop at depth 1 containing: %outer.header<header>,%early.exit.outer,%outer.latch<latch>,%outer.latch.loopexit,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.check<exiting>,%vector.early.exit.1
; CHECK-NEXT:     Loop at depth 2 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  br label %outer.header

outer.header:
  br label %inner.header

inner.header:
  %iv = phi i64 [ 0, %outer.header ], [ %iv.next, %inner.latch ]
  br i1 %c, label %early.exit.leave, label %inner.body

inner.body:
  %gep = getelementptr inbounds i8, ptr %src, i64 %iv
  %ld = load i8, ptr %gep, align 1
  %cmp = icmp eq i8 %ld, 0
  br i1 %cmp, label %early.exit.outer, label %inner.latch

inner.latch:
  %iv.next = add i64 %iv, 1
  %exit.cond = icmp ult i64 %iv, 63
  br i1 %exit.cond, label %inner.header, label %outer.latch

outer.latch:
  br label %outer.header

early.exit.outer:
  br label %outer.latch

early.exit.leave:
  ret i32 0
}

; Tests when an inner loop has two early exits at different loop levels
; (one going to a middle loop, one going to the outer loop).
define i32 @multi_early_exit_two_different_loops(i1 %c, ptr dereferenceable(1024) %src) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_two_different_loops':
; CHECK-NEXT: Loop at depth 1 containing: %outer.header<header>,%middle.header,%early.exit.outer,%early.exit.middle,%middle.latch,%outer.latch<latch>,%middle.latch.loopexit,%outer.latch.loopexit,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.check,%vector.early.exit.1,%vector.early.exit.0
; CHECK-NEXT:     Loop at depth 2 containing: %middle.header<header>,%early.exit.middle,%middle.latch<latch><exiting>,%middle.latch.loopexit,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.check<exiting>,%vector.early.exit.0
; CHECK-NEXT:         Loop at depth 3 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
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
  %gep = getelementptr inbounds i8, ptr %src, i64 %iv
  %ld = load i8, ptr %gep, align 1
  %cmp = icmp eq i8 %ld, 0
  br i1 %cmp, label %early.exit.outer, label %inner.latch

inner.latch:
  %iv.next = add i64 %iv, 1
  %exit.cond = icmp ult i64 %iv, 63
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
define i32 @multi_early_exit_two_different_loops_reversed(i1 %c, ptr dereferenceable(1024) %src) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_two_different_loops_reversed':
; CHECK-NEXT: Loop at depth 1 containing: %outer.header<header>,%middle.header,%early.exit.middle,%middle.latch,%early.exit.outer,%outer.latch<latch>,%middle.latch.loopexit,%outer.latch.loopexit,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.check,%vector.early.exit.1,%vector.early.exit.0
; CHECK-NEXT:     Loop at depth 2 containing: %middle.header<header>,%early.exit.middle,%middle.latch<latch><exiting>,%middle.latch.loopexit,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.check<exiting>,%vector.early.exit.1
; CHECK-NEXT:         Loop at depth 3 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
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
  %gep = getelementptr inbounds i8, ptr %src, i64 %iv
  %ld = load i8, ptr %gep, align 1
  %cmp = icmp eq i8 %ld, 0
  br i1 %cmp, label %early.exit.middle, label %inner.latch

inner.latch:
  %iv.next = add i64 %iv, 1
  %exit.cond = icmp ult i64 %iv, 63
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

; Tests when an early-exit inner loop has 3 early exits where the
; intermediate exit-routing blocks have multiple successors (i.e., the
; vector.early.exit.check block itself has >1 successor), and all exits leave
; all loops.
define i32 @multi_early_exit_all_leave_with_multi_succ_exit_check(i1 %c.1, i1 %c.2, ptr dereferenceable(1024) %src) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_all_leave_with_multi_succ_exit_check':
; CHECK-NEXT: Loop at depth 1 containing: %outer.header<header>,%outer.latch<latch>,%vector.ph,%vector.body<exiting>,%vector.body.interim,%middle.block
; CHECK-NEXT:     Loop at depth 2 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  br label %outer.header

outer.header:
  br label %inner.header

inner.header:
  %iv = phi i64 [ 0, %outer.header ], [ %iv.next, %inner.latch ]
  br i1 %c.1, label %inner.body, label %early.exit.1

inner.body:
  br i1 %c.2, label %inner.body.2, label %early.exit.2

inner.body.2:
  %gep = getelementptr inbounds i8, ptr %src, i64 %iv
  %ld = load i8, ptr %gep, align 1
  %cmp = icmp eq i8 %ld, 0
  br i1 %cmp, label %early.exit.3, label %inner.latch

inner.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %exit.cond = icmp samesign ult i64 %iv, 63
  br i1 %exit.cond, label %inner.header, label %outer.latch

outer.latch:
  br label %outer.header

early.exit.1:
  ret i32 0

early.exit.2:
  ret i32 1

early.exit.3:
  ret i32 2
}

; Tests with 3 levels of loop nesting (outer/middle/inner) where the inner loop
; has 3 early exits that ALL leave ALL loops. The exit routing blocks have
; multiple successors so connectToPredecessors places them in the vectorized
; loop. The fixup pass must move them all the way out (not just one level up).
define i32 @multi_early_exit_all_leave_three_nested_loops(i1 %c.1, i1 %c.2, ptr dereferenceable(1024) %src) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_all_leave_three_nested_loops':
; CHECK-NEXT: Loop at depth 1 containing: %outer.header<header>,%middle.header,%middle.latch,%outer.latch<latch>,%vector.ph,%vector.body<exiting>,%vector.body.interim,%middle.block
; CHECK-NEXT:     Loop at depth 2 containing: %middle.header<header>,%middle.latch<latch><exiting>,%vector.ph,%vector.body<exiting>,%vector.body.interim,%middle.block
; CHECK-NEXT:         Loop at depth 3 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  br label %outer.header

outer.header:
  br label %middle.header

middle.header:
  br label %inner.header

inner.header:
  %iv = phi i64 [ 0, %middle.header ], [ %iv.next, %inner.latch ]
  br i1 %c.1, label %inner.body, label %early.exit.1

inner.body:
  br i1 %c.2, label %inner.body.2, label %early.exit.2

inner.body.2:
  %gep = getelementptr inbounds i8, ptr %src, i64 %iv
  %ld = load i8, ptr %gep, align 1
  %cmp = icmp eq i8 %ld, 0
  br i1 %cmp, label %early.exit.3, label %inner.latch

inner.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %exit.cond = icmp samesign ult i64 %iv, 63
  br i1 %exit.cond, label %inner.header, label %middle.latch

middle.latch:
  br i1 %c.1, label %middle.header, label %outer.latch

outer.latch:
  br label %outer.header

early.exit.1:
  ret i32 0

early.exit.2:
  ret i32 1

early.exit.3:
  ret i32 2
}

; Tests that dispatch blocks are correctly placed in the outermost loop when
; the innermost loop has two uncountable early exits at different loop depths:
; one to the outer loop and one outside all loops.
define i64 @multi_early_exit_to_outer_and_outside(ptr dereferenceable(1024) %p1, ptr dereferenceable(1024) %p2, i1 %c) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_to_outer_and_outside':
; CHECK-NEXT: Loop at depth 1 containing: %loop.outer<header>,%loop.middle,%loop.inner.end,%loop.middle.end,%loop.inner.found,%loop.outer.latch<latch><exiting>,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.0,%vector.early.exit.check<exiting>
; CHECK-NEXT:    Loop at depth 2 containing: %loop.middle<header>,%loop.inner.end<latch><exiting>,%vector.ph,%vector.body<exiting>,%vector.body.interim,%middle.block
; CHECK-NEXT:        Loop at depth 3 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  br label %loop.outer

loop.outer:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %loop.outer.latch ]
  br label %loop.middle

loop.middle:
  br label %loop.inner

loop.inner:
  %iv = phi i64 [ 0, %loop.middle ], [ %iv.next, %loop.inner.inc ]
  %gep1 = getelementptr inbounds i8, ptr %p1, i64 %iv
  %ld1 = load i8, ptr %gep1, align 1
  %cmp.early1 = icmp eq i8 %ld1, 0
  br i1 %cmp.early1, label %loop.inner.found, label %loop.inner.body

loop.inner.body:
  %gep2 = getelementptr inbounds i8, ptr %p2, i64 %iv
  %ld2 = load i8, ptr %gep2, align 1
  %cmp.early2 = icmp eq i8 %ld2, 0
  br i1 %cmp.early2, label %exit, label %loop.inner.inc

loop.inner.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1024
  br i1 %exitcond, label %loop.inner.end, label %loop.inner

loop.inner.found:
  br label %loop.outer.latch

loop.inner.end:
  br i1 %c, label %loop.middle, label %loop.middle.end

loop.middle.end:
  br label %loop.outer.latch

loop.outer.latch:
  %res = phi i64 [ 0, %loop.middle.end ], [ 42, %loop.inner.found ]
  %outer.iv.next = add i64 %outer.iv, 1
  %outer.cond = icmp eq i64 %outer.iv.next, 100
  br i1 %outer.cond, label %loop.outer.end, label %loop.outer

loop.outer.end:
  ret i64 %res

exit:
  ret i64 1
}

; Same as above but with early exits reversed: the first early exit goes outside
; all loops and the second goes to the outer loop.
define i64 @multi_early_exit_to_outside_and_outer(ptr dereferenceable(1024) %p1, ptr dereferenceable(1024) %p2, i1 %c) {
; CHECK-LABEL: Loop info for function 'multi_early_exit_to_outside_and_outer':
; CHECK-NEXT: Loop at depth 1 containing: %loop.outer<header>,%loop.middle,%loop.inner.end,%loop.middle.end,%loop.inner.found,%loop.outer.latch<latch><exiting>,%vector.ph,%vector.body,%vector.body.interim,%middle.block,%vector.early.exit.1,%vector.early.exit.check<exiting>
; CHECK-NEXT:    Loop at depth 2 containing: %loop.middle<header>,%loop.inner.end<latch><exiting>,%vector.ph,%vector.body<exiting>,%vector.body.interim,%middle.block
; CHECK-NEXT:        Loop at depth 3 containing: %vector.body<header><exiting>,%vector.body.interim<latch><exiting>
entry:
  br label %loop.outer

loop.outer:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %loop.outer.latch ]
  br label %loop.middle

loop.middle:
  br label %loop.inner

loop.inner:
  %iv = phi i64 [ 0, %loop.middle ], [ %iv.next, %loop.inner.inc ]
  %gep1 = getelementptr inbounds i8, ptr %p1, i64 %iv
  %ld1 = load i8, ptr %gep1, align 1
  %cmp.early1 = icmp eq i8 %ld1, 0
  br i1 %cmp.early1, label %exit, label %loop.inner.body

loop.inner.body:
  %gep2 = getelementptr inbounds i8, ptr %p2, i64 %iv
  %ld2 = load i8, ptr %gep2, align 1
  %cmp.early2 = icmp eq i8 %ld2, 0
  br i1 %cmp.early2, label %loop.inner.found, label %loop.inner.inc

loop.inner.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1024
  br i1 %exitcond, label %loop.inner.end, label %loop.inner

loop.inner.found:
  br label %loop.outer.latch

loop.inner.end:
  br i1 %c, label %loop.middle, label %loop.middle.end

loop.middle.end:
  br label %loop.outer.latch

loop.outer.latch:
  %res = phi i64 [ 0, %loop.middle.end ], [ 42, %loop.inner.found ]
  %outer.iv.next = add i64 %outer.iv, 1
  %outer.cond = icmp eq i64 %outer.iv.next, 100
  br i1 %outer.cond, label %loop.outer.end, label %loop.outer

loop.outer.end:
  ret i64 %res

exit:
  ret i64 1
}
