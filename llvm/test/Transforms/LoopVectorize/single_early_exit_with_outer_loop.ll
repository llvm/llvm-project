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
