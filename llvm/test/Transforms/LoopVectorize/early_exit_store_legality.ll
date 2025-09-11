; REQUIRES: asserts
; RUN: opt -S < %s -p loop-vectorize -debug-only=loop-vectorize -force-vector-width=4 -disable-output 2>&1 | FileCheck %s

define i64 @loop_contains_store(ptr %dest) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store'
; CHECK:       LV: Not vectorizing: Early exit loop with store but no supported condition load.
entry:
  %p1 = alloca [1024 x i8]
  call void @init_mem(ptr %p1, i64 1024)
  br label %loop

loop:
  %index = phi i64 [ %index.next, %loop.inc ], [ 3, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %p1, i64 %index
  %ld1 = load i32, ptr %arrayidx, align 1
  %arrayidx2 = getelementptr inbounds i32, ptr %dest, i64 %index
  store i32 %ld1, ptr %arrayidx2, align 4
  %cmp = icmp eq i32 %ld1, 1
  br i1 %cmp, label %loop.inc, label %loop.end

loop.inc:
  %index.next = add i64 %index, 1
  %exitcond = icmp ne i64 %index.next, 67
  br i1 %exitcond, label %loop, label %loop.end

loop.end:
  %retval = phi i64 [ %index, %loop ], [ 67, %loop.inc ]
  ret i64 %retval
}

define void @loop_contains_store_condition_load_has_single_user(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_condition_load_has_single_user'
; CHECK:       LV: Not vectorizing: Writes to memory unsupported in early exit loops.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_ee_condition_is_invariant(ptr dereferenceable(40) noalias %array, i16 %ee.val) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_ee_condition_is_invariant'
; CHECK:       LV: Not vectorizing: Early exit loop with store but no supported condition load.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_fcmp_condition(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_fcmp_condition'
; CHECK:       LV: Not vectorizing: Early exit loop with store but no supported condition load.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw half, ptr %pred, i64 %iv
  %ee.val = load half, ptr %ee.addr, align 2
  %ee.cond = fcmp ugt half %ee.val, 500.0
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_safe_dependency(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(96) %pred) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_safe_dependency'
; CHECK:       LV: Not vectorizing: Cannot determine whether critical uncountable exit load address does not alias with a memory write.
entry:
  %pred.plus.8 = getelementptr inbounds nuw i16, ptr %pred, i64 8
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred.plus.8, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  %some.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  store i16 42, ptr %some.addr, align 2
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_unsafe_dependency(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(80) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_unsafe_dependency'
; CHECK:       LV: Not vectorizing: Loop may fault.
entry:
  %unknown.offset = call i64 @get_an_unknown_offset()
  %unknown.cmp = icmp ult i64 %unknown.offset, 20
  %clamped.offset = select i1 %unknown.cmp, i64 %unknown.offset, i64 20
  %unknown.base = getelementptr i16, ptr %pred, i64 %clamped.offset
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %unknown.base, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  %some.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  store i16 42, ptr %some.addr, align 2
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_assumed_bounds(ptr noalias %array, ptr readonly %pred, i32 %n) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_assumed_bounds'
; CHECK:       LV: Not vectorizing: Loop may fault.
entry:
  %n_bytes = mul nuw nsw i32 %n, 2
  call void @llvm.assume(i1 true) [ "align"(ptr %pred, i64 2), "dereferenceable"(ptr %pred, i32 %n_bytes) ]
  %tc = sext i32 %n to i64
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, %tc
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_to_pointer_with_no_deref_info(ptr align 2 dereferenceable(40) readonly %load.array, ptr align 2 noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_to_pointer_with_no_deref_info'
; CHECK:       LV: Not vectorizing: Writes to memory unsupported in early exit loops.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %ld.addr = getelementptr inbounds nuw i16, ptr %load.array, i64 %iv
  %data = load i16, ptr %ld.addr, align 2
  %inc = add nsw i16 %data, 1
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_unknown_bounds(ptr align 2 dereferenceable(100) noalias %array, ptr align 2 dereferenceable(100) readonly %pred, i64 %n) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_unknown_bounds'
; CHECK:       LV: Not vectorizing: Loop may fault.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, %n
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_volatile(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_volatile'
; CHECK:       LV: Not vectorizing: Complex writes to memory unsupported in early exit loops.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store volatile i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_to_invariant_location(ptr dereferenceable(40) readonly %array, ptr align 2 dereferenceable(40) readonly %pred, ptr noalias %store_addr) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_to_invariant_location'
; CHECK:       LV: Not vectorizing: Writes to memory unsupported in early exit loops.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %store_addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_in_latch_block(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_in_latch_block'
; CHECK:       LV: Not vectorizing: Writes to memory unsupported in early exit loops.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  store i16 %inc, ptr %st.addr, align 2
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_requiring_alias_check(ptr dereferenceable(40) %array, ptr align 2 dereferenceable(40) %pred) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_requiring_alias_check'
; CHECK:       LV: Not vectorizing: Cannot determine whether critical uncountable exit load address does not alias with a memory write.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_condition_load_is_chained(ptr dereferenceable(40) noalias %array, ptr align 8 dereferenceable(160) readonly %offsets, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_condition_load_is_chained'
; CHECK:       LV: Not vectorizing: Uncountable exit condition depends on load with an address that is not an add recurrence in the loop.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %gather.addr = getelementptr inbounds nuw i64, ptr %offsets, i64 %iv
  %ee.offset = load i64, ptr %gather.addr, align 8
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %ee.offset
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_decrementing_iv(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_decrementing_iv'
; CHECK:       LV: Not vectorizing: Writes to memory unsupported in early exit loops.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 19, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = sub nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 0
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_condition_load_requires_gather(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(512) readonly %pred, ptr align 1 dereferenceable(20) readonly %offsets) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_condition_load_requires_gather'
; CHECK:       LV: Not vectorizing: Uncountable exit condition depends on load with an address that is not an add recurrence in the loop.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %offset.addr = getelementptr inbounds nuw i8, ptr %offsets, i64 %iv
  %offset = load i8, ptr %offset.addr, align 1
  %offset.zext = zext i8 %offset to i64
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %offset.zext
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_uncounted_exit_is_a_switch(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_uncounted_exit_is_a_switch'
; CHECK:       LV: Not vectorizing: Loop contains an unsupported switch
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  switch i16 %ee.val, label %for.inc [ i16 500, label %exit ]

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @loop_contains_store_uncounted_exit_is_not_guaranteed_to_execute(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'loop_contains_store_uncounted_exit_is_not_guaranteed_to_execute'
; CHECK:       LV: Not vectorizing: Early exit is not the latch predecessor.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %rem = urem i64 %iv, 5
  %skip.ee.cmp = icmp eq i64 %rem, 0
  br i1 %skip.ee.cmp, label %for.inc, label %ee.block

ee.block:
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @test_nodep(ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'test_nodep'
; CHECK:       LV: Not vectorizing: Cannot determine whether critical uncountable exit load address does not alias with a memory write.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  store i16 0, ptr %st.addr, align 2
  %ee.val = load i16, ptr %st.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @histogram_with_uncountable_exit(ptr noalias %buckets, ptr readonly %indices, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'histogram_with_uncountable_exit'
; CHECK:       LV: Not vectorizing: Cannot vectorize unsafe dependencies in uncountable exit loop with side effects.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %gep.indices = getelementptr inbounds i32, ptr %indices, i64 %iv
  %l.idx = load i32, ptr %gep.indices, align 4
  %idxprom1 = zext i32 %l.idx to i64
  %gep.bucket = getelementptr inbounds i32, ptr %buckets, i64 %idxprom1
  %l.bucket = load i32, ptr %gep.bucket, align 4
  %inc = add nsw i32 %l.bucket, 1
  store i32 %inc, ptr %gep.bucket, align 4
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @uncountable_exit_condition_address_is_invariant(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(2) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'uncountable_exit_condition_address_is_invariant'
; CHECK:       LV: Not vectorizing: Uncountable exit condition depends on load with an address that is not an add recurrence in the loop.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.val = load i16, ptr %pred, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

define void @uncountable_exit_condition_address_is_addrec_in_outer_loop(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(2) readonly %pred) {
; CHECK-LABEL: LV: Checking a loop in 'uncountable_exit_condition_address_is_addrec_in_outer_loop'
; CHECK:       LV: Not vectorizing: Uncountable exit condition depends on load with an address that is not an add recurrence in the loop.
entry:
  br label %outer.body

outer.body:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %outer.inc ]
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %outer.iv
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %outer.body ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %outer.inc, label %for.body

outer.inc:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %outer.cond = icmp eq i64 %outer.iv.next, 2
  br i1 %outer.cond, label %exit, label %outer.body

exit:
  ret void
}

declare void @init_mem(ptr, i64);
declare i64 @get_an_unknown_offset();
