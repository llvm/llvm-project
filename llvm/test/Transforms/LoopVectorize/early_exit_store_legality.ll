; REQUIRES: asserts
; RUN: opt -S < %s -p loop-vectorize -debug-only=loop-vectorize -enable-early-exit-vectorization-with-side-effects \
; RUN:   -force-vector-width=4 -disable-output 2>&1 | FileCheck %s --check-prefix=CHECK-DEBUG
; RUN: opt -S < %s -p loop-vectorize -pass-remarks-analysis='loop-vectorize' -enable-early-exit-vectorization-with-side-effects \
; RUN:   -force-vector-width=4 -disable-output 2>&1 | FileCheck %s --check-prefix=CHECK-REMARK

;; This currently doesn't vectorize because the load used to determine the
;; uncountable exit condition has a second user (the store).
define i64 @loop_contains_store(ptr dereferenceable(1024) %p1, ptr noalias %dest) !dbg !10 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store'
; CHECK-DEBUG:       LV: Not vectorizing: Early exit loop with store but no supported condition load.
; CHECK-REMARK:      foo.c:10:3: loop not vectorized: Early exit loop with store but no supported condition load
entry:
  br label %loop, !dbg !11

loop:
  %index = phi i64 [ %index.next, %loop.inc ], [ 3, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %p1, i64 %index
  %ld1 = load i32, ptr %arrayidx, align 1
  %arrayidx2 = getelementptr inbounds i32, ptr %dest, i64 %index
  store i32 %ld1, ptr %arrayidx2, align 4
  %cmp = icmp eq i32 %ld1, 1
  br i1 %cmp, label %loop.inc, label %loop.end, !dbg !11

loop.inc:
  %index.next = add i64 %index, 1
  %exitcond = icmp ne i64 %index.next, 67
  br i1 %exitcond, label %loop, label %loop.end, !dbg !11

loop.end:
  %retval = phi i64 [ %index, %loop ], [ 67, %loop.inc ]
  ret i64 %retval
}

define void @loop_contains_store_condition_load_has_single_user(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_condition_load_has_single_user'
; CHECK-DEBUG:       LV: We can vectorize this loop!
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

define void @loop_contains_store_multidim_condition_load(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(80) readonly %pred) !dbg !12 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_multidim_condition_load'
; CHECK-DEBUG:       LV: Not vectorizing: Unable to determine early exit condition for loop with side effects.
; CHECK-REMARK:      foo.c:20:3: loop not vectorized: Unable to determine early exit condition for loop with side effects
entry:
  br label %for.body, !dbg !13

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw [2 x i16], ptr %pred, i64 %iv, i64 0
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !13

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !13

exit:
  ret void
}

;; Exit-condition load on the RHS of the icmp must still be accepted.
define void @swapped_cmp_operands(ptr noalias %array, ptr %pred) {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'swapped_cmp_operands'
; CHECK-DEBUG:       LV: We can vectorize this loop!
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %st.addr = getelementptr i16, ptr %array, i64 %iv
  store i16 0, ptr %st.addr, align 2
  %ee.addr = getelementptr i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp slt i16 500, %ee.val
  br i1 %ee.cond, label %exit, label %latch

latch:
  %iv.next = add i64 %iv, 1
  %latch.cond = icmp eq i64 %iv.next, 20
  br i1 %latch.cond, label %exit, label %loop

exit:
  ret void
}

;; Avoid vectorization because we will either exit on the first iteration, or
;; never exit early.
;; We shouldn't see IR like this if LV-LICM has done its job.
define void @novec_loop_contains_store_ee_condition_is_invariant(ptr dereferenceable(40) noalias %array, i16 %ee.val) !dbg !16 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'novec_loop_contains_store_ee_condition_is_invariant'
; CHECK-DEBUG:       LV: Not vectorizing: Early exit loop with store but no supported condition load.
; CHECK-REMARK:      foo.c:40:3: loop not vectorized: Early exit loop with store but no supported condition load
entry:
  br label %for.body, !dbg !17

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !17

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !17

exit:
  ret void
}

;; Vectorizeable, needs work on exit condition recipe collection.
define void @loop_contains_store_fcmp_condition(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) !dbg !18 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_fcmp_condition'
; CHECK-DEBUG:       LV: Not vectorizing: Early exit loop with store but no supported condition load.
; CHECK-REMARK:      foo.c:50:3: loop not vectorized: Early exit loop with store but no supported condition load
entry:
  br label %for.body, !dbg !19

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw half, ptr %pred, i64 %iv
  %ee.val = load half, ptr %ee.addr, align 2
  %ee.cond = fcmp ugt half %ee.val, 500.0
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !19

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !19

exit:
  ret void
}

;; Vectorizeable, needs work on alias checks for the exit condition load.
define void @loop_contains_store_safe_dependency(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(96) %pred) !dbg !20 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_safe_dependency'
; CHECK-DEBUG:       LV: Not vectorizing: Cannot determine whether critical uncountable exit load address does not alias with a memory write.
; CHECK-REMARK:      foo.c:60:3: loop not vectorized: Cannot determine whether critical uncountable exit load address does not alias with a memory write
entry:
  %pred.plus.8 = getelementptr inbounds nuw i16, ptr %pred, i64 8
  br label %for.body, !dbg !21

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
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !21

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !21

exit:
  ret void
}

;; Possibly vectorizeable, but would require some runtime checks.
define void @loop_contains_store_unsafe_dependency(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(80) %pred) !dbg !22 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_unsafe_dependency'
; CHECK-DEBUG:       LV: Not vectorizing: Cannot determine whether critical uncountable exit load address does not alias with a memory write.
; CHECK-REMARK:      foo.c:70:3: loop not vectorized: Cannot determine whether critical uncountable exit load address does not alias with a memory write
entry:
  %unknown.offset = call i64 @get_an_unknown_offset()
  %unknown.cmp = icmp ult i64 %unknown.offset, 20
  %clamped.offset = select i1 %unknown.cmp, i64 %unknown.offset, i64 20
  %unknown.base = getelementptr i16, ptr %pred, i64 %clamped.offset
  br label %for.body, !dbg !23

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
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !23

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !23

exit:
  ret void
}

;; Vectorizeable, needs runtime checks to determine whether the iteration count
;; might exceed known dereferenceable extents.
;; Alternatively, we could use masked.load.ff or vp.load.ff
define void @loop_contains_store_assumed_bounds(ptr noalias %array, ptr readonly %pred, i64 %n) {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_assumed_bounds'
; CHECK-DEBUG:       LV: We can vectorize this loop!
entry:
  %n_bytes = mul nuw nsw i64 %n, 2
  call void @llvm.assume(i1 true) [ "align"(ptr %pred, i64 2), "dereferenceable"(ptr %pred, i64 %n_bytes) ]
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

define void @loop_contains_store_to_pointer_with_no_deref_info(ptr align 2 dereferenceable(40) readonly %load.array, ptr align 2 noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_to_pointer_with_no_deref_info'
; CHECK-DEBUG:       LV: We can vectorize this loop!
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

;; Vectorizeable, requires runtime checks and/or ff loads.
define void @loop_contains_store_unknown_bounds(ptr align 2 dereferenceable(100) noalias %array, ptr align 2 dereferenceable(100) readonly %pred, i64 %n) {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_unknown_bounds'
; CHECK-DEBUG:       LV: We can vectorize this loop!
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

;; Avoid vectorization, volatile memory locations may have unexpected behaviour
;; if we try to vectorize.
define void @novec_loop_contains_store_volatile(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) !dbg !30 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'novec_loop_contains_store_volatile'
; CHECK-DEBUG:       LV: Not vectorizing: Complex writes to memory unsupported in early exit loops.
; CHECK-REMARK:      foo.c:110:3: loop not vectorized: Cannot vectorize early exit loop with complex writes to memory
; CHECK-REMARK-NEXT: foo.c:110:3: loop not vectorized: instruction cannot be vectorized
entry:
  br label %for.body, !dbg !31

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store volatile i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !31

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !31

exit:
  ret void
}

;; Vectorizeable, but we really want LICM to sink the store out of the loop
define void @loop_contains_store_to_invariant_location(ptr dereferenceable(40) readonly %array, ptr align 2 dereferenceable(40) readonly %pred, ptr noalias %store_addr) !dbg !32 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_to_invariant_location'
; CHECK-DEBUG:       LV: Not vectorizing: Cannot vectorize early exit loops with stores to loop-invariant addresses.
; CHECK-REMARK:      foo.c:120:3: loop not vectorized: Cannot vectorize early exit loops with stores to loop-invariant addresses
entry:
  br label %for.body, !dbg !33

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %store_addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !33

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !33

exit:
  ret void
}

define void @loop_contains_store_in_latch_block(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_in_latch_block'
; CHECK-DEBUG:       LV: We can vectorize this loop!
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

;; Vectorizeable, requires runtime checks.
define void @loop_contains_store_requiring_alias_check(ptr dereferenceable(40) %array, ptr align 2 dereferenceable(40) %pred) !dbg !36 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_requiring_alias_check'
; CHECK-DEBUG:       LV: Not vectorizing: Cannot determine whether critical uncountable exit load address does not alias with a memory write.
; CHECK-REMARK:      foo.c:140:3: loop not vectorized: Cannot determine whether critical uncountable exit load address does not alias with a memory write
entry:
  br label %for.body, !dbg !37

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !37

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !37

exit:
  ret void
}

define void @loop_contains_store_decrementing_iv(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) !dbg !38 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_decrementing_iv'
; CHECK-DEBUG:       LV: Not vectorizing: Early exit loop with side effects contains load used by the exit condition with an unsupported memory access pattern
; CHECK-REMARK:      foo.c:150:3: loop not vectorized: Early exit loop with side effects contains load used by the exit condition with an unsupported memory access pattern
entry:
  br label %for.body, !dbg !39

for.body:
  %iv = phi i64 [ 19, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !39

for.inc:
  %iv.next = sub nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 0
  br i1 %counted.cond, label %exit, label %for.body, !dbg !39

exit:
  ret void
}

;; Vectorizeable, requires improvements to exit condition recipe collection and
;; masked.gather.ff
define void @loop_contains_store_condition_load_requires_gather(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(512) readonly %pred, ptr align 1 dereferenceable(20) readonly %offsets) !dbg !40 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_condition_load_requires_gather'
; CHECK-DEBUG:       LV: Not vectorizing: Uncountable exit condition depends on load with an address that is not an add recurrence in the loop.
; CHECK-REMARK:      foo.c:160:3: loop not vectorized: Uncountable exit condition depends on load with an address that is not an add recurrence in the loop
entry:
  br label %for.body, !dbg !41

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
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !41

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !41

exit:
  ret void
}

;; Vectorizeable, requires improvements in handling switch instructions in LV.
define void @loop_contains_store_uncounted_exit_is_a_switch(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) !dbg !42 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_uncounted_exit_is_a_switch'
; CHECK-DEBUG:       LV: Not vectorizing: Loop contains an unsupported switch
; CHECK-REMARK:      foo.c:170:3: loop not vectorized: Loop contains an unsupported switch
; CHECK-REMARK-NEXT: foo.c:170:3: loop not vectorized: Early exit loop contains operations that cannot be speculatively executed
entry:
  br label %for.body, !dbg !43

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  switch i16 %ee.val, label %for.inc [ i16 500, label %exit ], !dbg !43

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !43

exit:
  ret void
}

;; Vectorizeable, needs to extend the predicated early exit work and improve
;; exit condition recipe collection.
define void @loop_contains_store_uncounted_exit_is_not_guaranteed_to_execute(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) !dbg !44 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_uncounted_exit_is_not_guaranteed_to_execute'
; CHECK-DEBUG:       LV: Not vectorizing: Load for uncountable exit not guaranteed to execute.
; CHECK-REMARK:      foo.c:180:3: loop not vectorized: Load for uncountable exit not guaranteed to execute
entry:
  br label %for.body, !dbg !45

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %rem = urem i64 %iv, 5
  %skip.ee.cmp = icmp eq i64 %rem, 0
  br i1 %skip.ee.cmp, label %for.inc, label %ee.block, !dbg !45

ee.block:
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !45

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !45

exit:
  ret void
}

;; Vectorizeable, requires better alias checking in legality. However, hopefully
;; we wouldn't get this as input IR since it stores to the same address that
;; we load from immediately afterwards.
define void @test_nodep(ptr align 2 dereferenceable(40) readonly %pred) !dbg !46 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'test_nodep'
; CHECK-DEBUG:       LV: Not vectorizing: Cannot determine whether critical uncountable exit load address does not alias with a memory write.
; CHECK-REMARK:      foo.c:190:3: loop not vectorized: Cannot determine whether critical uncountable exit load address does not alias with a memory write
entry:
  br label %for.body, !dbg !47

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  store i16 0, ptr %st.addr, align 2
  %ee.val = load i16, ptr %st.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !47

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !47

exit:
  ret void
}

;; Vectorizeable, requires working with the existing histogram code.
define void @histogram_with_uncountable_exit(ptr noalias %buckets, ptr readonly %indices, ptr align 2 dereferenceable(40) readonly %pred) !dbg !48 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'histogram_with_uncountable_exit'
; CHECK-DEBUG:       LV: Not vectorizing: Cannot vectorize unsafe dependencies in uncountable exit loop with side effects.
; CHECK-REMARK:      foo.c:200:3: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma clang loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
; CHECK-REMARK-NEXT: Unsafe indirect dependence.
; CHECK-REMARK-NEXT: foo.c:200:3: loop not vectorized: Cannot vectorize unsafe dependencies in uncountable exit loop with side effects
entry:
  br label %for.body, !dbg !49

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
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !49

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !49

exit:
  ret void
}

;; Vectorizeable, requires processing more than one exit.
define void @loop_contains_store_between_two_early_exits(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred, ptr align 2 dereferenceable(40) readonly %pred2) !dbg !50 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_between_two_early_exits'
; CHECK-DEBUG:       LV: Not vectorizing: Load for uncountable exit not guaranteed to execute.
; CHECK-REMARK:      foo.c:210:3: loop not vectorized: Load for uncountable exit not guaranteed to execute
entry:
  br label %for.body, !dbg !51

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp slt i16 %ee.val, 250
  br i1 %ee.cond, label %exit, label %for.cont, !dbg !51

for.cont:
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee2.addr = getelementptr inbounds nuw i16, ptr %pred2, i64 %iv
  %ee2.val = load i16, ptr %ee2.addr, align 2
  %ee2.cond = icmp sgt i16 %ee2.val, 500
  br i1 %ee2.cond, label %exit, label %for.inc, !dbg !51

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !51

exit:
  ret void
}

;; Vectorizeable, requires processing more than one exit.
define void @loop_contains_store_before_two_early_exits(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred, ptr align 2 dereferenceable(40) readonly %pred2) !dbg !52 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'loop_contains_store_before_two_early_exits'
; CHECK-DEBUG:       LV: Not vectorizing: Load for uncountable exit not guaranteed to execute.
; CHECK-REMARK:      foo.c:220:3: loop not vectorized: Load for uncountable exit not guaranteed to execute
entry:
  br label %for.body, !dbg !53

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp slt i16 %ee.val, 250
  br i1 %ee.cond, label %exit, label %for.cont, !dbg !53

for.cont:
  %ee2.addr = getelementptr inbounds nuw i16, ptr %pred2, i64 %iv
  %ee2.val = load i16, ptr %ee2.addr, align 2
  %ee2.cond = icmp sgt i16 %ee2.val, 500
  br i1 %ee2.cond, label %exit, label %for.inc, !dbg !53

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !53

exit:
  ret void
}

;; Vectorizeable, requires processing more than one exit.
define void @one_uncountable_two_countable_exits(ptr dereferenceable(1024) noalias %array, ptr dereferenceable(1024) readonly %pred) !dbg !54 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'one_uncountable_two_countable_exits'
; CHECK-DEBUG:       LV: Not vectorizing: Load for uncountable exit not guaranteed to execute.
; CHECK-REMARK:      foo.c:230:3: loop not vectorized: Load for uncountable exit not guaranteed to execute
entry:
  br label %loop, !dbg !55

loop:
  %iv = phi i64 [ %iv.next, %loop.inc ], [ 3, %entry ]
  %ce.ee.cmp = icmp ne i64 %iv, 64
  br i1 %ce.ee.cmp, label %update, label %loop.end, !dbg !55

update:
  %st.addr = getelementptr inbounds i8, ptr %array, i64 %iv
  %data = load i8, ptr %st.addr, align 1
  %inc = add nsw i8 %data, 1
  store i8 %inc, ptr %st.addr, align 1
  %ee.addr = getelementptr inbounds i8, ptr %pred, i64 %iv
  %ee.val = load i8, ptr %ee.addr, align 1
  %ee.cond = icmp eq i8 %ee.val, 37
  br i1 %ee.cond, label %loop.end, label %loop.inc, !dbg !55

loop.inc:
  %iv.next = add i64 %iv, 1
  %ce.latch.cmp = icmp ne i64 %iv.next, 128
  br i1 %ce.latch.cmp, label %loop, label %loop.end, !dbg !55

loop.end:
  ret void
}

;; Vectorizeable, need to handle reductions.
define i16 @uncountable_exit_with_reduction(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) !dbg !56 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'uncountable_exit_with_reduction'
; CHECK-DEBUG:       LV: Not vectorizing: Found an unidentified PHI %rdx = phi i16 [ 0, %entry ], [ %rdx.next, %for.inc ]
; CHECK-REMARK:      foo.c:240:3: loop not vectorized: value that could not be identified as reduction is used outside the loop
entry:
  br label %for.body, !dbg !57

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %rdx = phi i16 [ 0, %entry ], [ %rdx.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !57

for.inc:
  %rdx.next = add i16 %rdx, %data
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !57

exit:
  %res = phi i16 [ %rdx, %for.body ], [ %rdx.next, %for.inc ]
  ret i16 %res
}

define i16 @uncountable_exit_with_live_out(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'uncountable_exit_with_live_out'
; CHECK-DEBUG:       LV: We can vectorize this loop!
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
  ret i16 %data
}

; Vectorizeable, requires improvements in dereferenceability checks
define void @uncountable_exit_with_constant_nonunit_stride(ptr dereferenceable(4000) noalias %array, ptr align 2 dereferenceable(4000) readonly %pred) !dbg !60 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'uncountable_exit_with_constant_nonunit_stride'
; CHECK-DEBUG:       LV: We can vectorize this loop!
; CHECK-DEBUG:       LV: Not vectorizing: unable to calculate the loop count due to complex control flow.
; CHECK-REMARK:      foo.c:260:3: loop not vectorized: unable to calculate the loop count due to complex control flow
entry:
  br label %for.body, !dbg !61

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !61

for.inc:
  %iv.next = add nuw nsw i64 %iv, 20
  %counted.cond = icmp slt i64 %iv.next, 2001
  br i1 %counted.cond, label %exit, label %for.body, !dbg !61

exit:
  ret void
}

; Vectorizeable, requires improvements in dereferenceability checks
define void @uncountable_exit_with_invariant_but_unknown_stride(ptr dereferenceable(4000) noalias %array, ptr align 2 dereferenceable(4000) readonly %pred, i64 %stride) !dbg !62 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'uncountable_exit_with_invariant_but_unknown_stride'
; CHECK-DEBUG:       LV: Not vectorizing: Cannot determine exact exit count for latch block.
; CHECK-REMARK:      foo.c:270:3: loop not vectorized: Cannot vectorize early exit loop
; CHECK-REMARK-NEXT: foo.c:270:3: loop not vectorized: could not determine number of loop iterations
entry:
  br label %for.body, !dbg !63

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !63

for.inc:
  %iv.next = add nuw nsw i64 %iv, %stride
  %counted.cond = icmp slt i64 %iv.next, 2001
  br i1 %counted.cond, label %exit, label %for.body, !dbg !63

exit:
  ret void
}

define i32 @uncountable_exit_with_separate_exit_block(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred) {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'uncountable_exit_with_separate_exit_block'
; CHECK-DEBUG:       LV: We can vectorize this loop!
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
  br i1 %ee.cond, label %exit.uncountable, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit.countable, label %for.body

exit.countable:
  ret i32 0

exit.uncountable:
  ret i32 1
}

; This loop passes legality checks, but fails in vplan due to unsupported
; getelementptr used for the critical load that feeds the exit condition.
define void @uncountable_exit_condition_load_offset_from_iv(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(42) readonly %pred) !dbg !64 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'uncountable_exit_condition_load_offset_from_iv'
; CHECK-DEBUG:       LV: Not vectorizing: Early exit loop with side effects contains load used by the exit condition with an unsupported memory access pattern
; CHECK-REMARK:      foo.c:280:3: loop not vectorized: Early exit loop with side effects contains load used by the exit condition with an unsupported memory access pattern
entry:
  br label %for.body, !dbg !65

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.offset = add nuw nsw i64 %iv, 1
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %ee.offset
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !65

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !65

exit:
  ret void
}

define i32 @uncountable_exit_with_masked_ldst_separate_condition(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(40) readonly %pred, ptr align 2 readonly %st.pred) !dbg !66 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'uncountable_exit_with_masked_ldst_separate_condition'
; CHECK-DEBUG:       LV: We can vectorize this loop!
; CHECK-DEBUG:       LV: Not vectorizing: Early exit loop with side effects contains unsupported conditional memory operations
; CHECK-DEBUG:       LV: Vectorization is possible but not beneficial.
; CHECK-REMARK:      foo.c:290:3: loop not vectorized: Early exit loop with side effects contains unsupported conditional memory operations
entry:
  br label %for.body, !dbg !67

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %stp.gep = getelementptr inbounds nuw i16, ptr %st.pred, i64 %iv
  %stp.val = load i16, ptr %stp.gep, align 2
  %stp.cond = icmp slt i16 %stp.val, 2345
  br i1 %stp.cond, label %ldst.block, label %ee.block, !dbg !67

ldst.block:
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  br label %ee.block, !dbg !67

ee.block:
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit.uncountable, label %for.inc, !dbg !67

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit.countable, label %for.body, !dbg !67

exit.countable:
  ret i32 0

exit.uncountable:
  ret i32 1
}

;; Avoid vectorization; similar to another invariant test above, we would either
;; exit immediately on the first lane or never take the early exit. Should be
;; versioned before reaching LV.
define void @novec_uncountable_exit_condition_address_is_invariant(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(2) readonly %pred) !dbg !68 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'novec_uncountable_exit_condition_address_is_invariant'
; CHECK-DEBUG:       LV: Not vectorizing: Uncountable exit condition depends on load with an address that is not an add recurrence in the loop.
; CHECK-REMARK:      foo.c:300:3: loop not vectorized: Uncountable exit condition depends on load with an address that is not an add recurrence in the loop
entry:
  br label %for.body, !dbg !69

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.val = load i16, ptr %pred, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !69

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !69

exit:
  ret void
}

;; Avoid vectorization; similar to the test above, we would either exit
;; immediately on the first lane or never take the early exit. Should be
;; versioned before reaching LV.
define void @novec_uncountable_exit_condition_address_is_addrec_in_outer_loop(ptr dereferenceable(40) noalias %array, ptr align 2 dereferenceable(2) readonly %pred) !dbg !70 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'novec_uncountable_exit_condition_address_is_addrec_in_outer_loop'
; CHECK-DEBUG:       LV: Not vectorizing: Uncountable exit condition depends on load with an address that is not an add recurrence in the loop.
; CHECK-REMARK:      foo.c:310:3: loop not vectorized: Uncountable exit condition depends on load with an address that is not an add recurrence in the loop
entry:
  br label %outer.body, !dbg !71

outer.body:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %outer.inc ]
  %ee.addr = getelementptr inbounds nuw i16, ptr %pred, i64 %outer.iv
  br label %for.body, !dbg !71

for.body:
  %iv = phi i64 [ 0, %outer.body ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !71

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %outer.inc, label %for.body, !dbg !71

outer.inc:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %outer.cond = icmp eq i64 %outer.iv.next, 2
  br i1 %outer.cond, label %exit, label %outer.body, !dbg !71

exit:
  ret void
}

;; ICE was caused by assert for the load used in the uncountable exit condition
;; being guaranteed to execute.
@ee.global = external global [4 x i8]
define void @crash_conditional_load_for_uncountable_exit(ptr dereferenceable(40) noalias %store.area) !dbg !72 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'crash_conditional_load_for_uncountable_exit'
; CHECK-DEBUG:       LV: Not vectorizing: Load for uncountable exit not guaranteed to execute.
; CHECK-REMARK:      foo.c:320:3: loop not vectorized: Load for uncountable exit not guaranteed to execute
entry:
  br label %for.body, !dbg !73

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %ee.addr = getelementptr i8, ptr @ee.global, i64 %iv
  br i1 false, label %ee.block, label %invalid.block, !dbg !73

ee.block:
  %ee.val = load i8, ptr %ee.addr, align 1
  store i16 0, ptr %store.area, align 2
  %ee.cmp = icmp eq i8 %ee.val, 0
  br i1 %ee.cmp, label %for.inc, label %invalid.block, !dbg !73

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 10
  br i1 %counted.cond, label %invalid.block, label %for.body, !dbg !73

invalid.block:
  unreachable
}

define void @crash_conditional_load_for_uncountable_exit_argptr(ptr dereferenceable(40) noalias %store.area, ptr dereferenceable(4) %load.area, i1 %skip.cond) !dbg !74 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'crash_conditional_load_for_uncountable_exit_argptr'
; CHECK-DEBUG:       LV: Not vectorizing: Early exit loop with store but no supported condition load.
; CHECK-REMARK:      foo.c:330:3: loop not vectorized: Early exit loop with store but no supported condition load
entry:
  br label %for.body, !dbg !75

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %ee.addr = getelementptr i8, ptr %load.area, i64 %iv
  br i1 %skip.cond, label %ee.block, label %invalid.block, !dbg !75

ee.block:
  %ee.val = load i8, ptr %ee.addr, align 1
  store i16 0, ptr %store.area, align 2
  %ee.cmp = icmp eq i8 %ee.val, 0
  br i1 %ee.cmp, label %for.inc, label %invalid.block, !dbg !75

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 10
  br i1 %counted.cond, label %invalid.block, label %for.body, !dbg !75

invalid.block:
  unreachable
}

define void @combined_exit_conditions(ptr align 4 dereferenceable(80) readonly %src, ptr align 4 dereferenceable(80) noalias %dst, ptr align 4 dereferenceable(80) readonly %pred) !dbg !76 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'combined_exit_conditions'
; CHECK-DEBUG:       LV:  Not vectorizing: Cannot vectorize uncountable loop.
; CHECK-REMARK:      foo.c:340:3: loop not vectorized: Cannot vectorize uncountable loop
entry:
  br label %for.body, !dbg !77

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %src.ptr = getelementptr inbounds nuw [4 x i8], ptr %src, i64 %iv
  %data = load i32, ptr %src.ptr, align 4
  %add = add nsw i32 %data, 1
  %dst.ptr = getelementptr inbounds nuw [4 x i8], ptr %dst, i64 %iv
  store i32 %add, ptr %dst.ptr, align 4
  %ee.ptr = getelementptr inbounds nuw [4 x i8], ptr %pred, i64 %iv
  %ee.val = load i32, ptr %ee.ptr, align 4
  %ee.cmp = icmp ne i32 %ee.val, 0
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cmp = icmp eq i64 %iv.next, 20
  %combined.cond = select i1 %ee.cmp, i1 true, i1 %counted.cmp
  br i1 %combined.cond, label %exit, label %for.body, !dbg !77

exit:
  ret void
}

define i64 @uncountable_exit_with_extra_induction(ptr dereferenceable(80) noalias %array, ptr align 4 dereferenceable(80) readonly %pred) !dbg !78 {
; CHECK-DEBUG-LABEL: LV: Checking a loop in 'uncountable_exit_with_extra_induction'
; CHECK-DEBUG:       LV: Not vectorizing: Early exit loop with side effects contains unsupported reductions, inductions or recurrences
; CHECK-REMARK:      foo.c:350:3: loop not vectorized: Early exit loop with side effects contains unsupported reductions, inductions or recurrences
entry:
  br label %for.body, !dbg !79

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %j = phi i64 [ 100, %entry ], [ %j.next, %for.inc ]
  %st.addr = getelementptr inbounds nuw i32, ptr %array, i64 %iv
  %j.trunc = trunc i64 %j to i32
  store i32 %j.trunc, ptr %st.addr, align 4
  %ee.addr = getelementptr inbounds nuw i32, ptr %pred, i64 %iv
  %ee.val = load i32, ptr %ee.addr, align 4
  %ee.cond = icmp sgt i32 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc, !dbg !79

for.inc:
  %j.next = add nuw nsw i64 %j, 2
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, 20
  br i1 %counted.cond, label %exit, label %for.body, !dbg !79

exit:
  %res = phi i64 [ %j, %for.body ], [ %j.next, %for.inc ]
  ret i64 %res
}


declare void @init_mem(ptr, i64);
declare i64 @get_an_unknown_offset();

!llvm.dbg.cu = !{!1000}
!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"PIC Level", i32 2}
!2 = !DIFile(filename: "foo.c", directory: "")
!3 = !DISubroutineType(types: !4)
!4 = !{}
!10 = distinct !DISubprogram(name: "loop_contains_store", scope: !2, file: !2, line: 10, type: !3, isLocal: false, isDefinition: true, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!11 = !DILocation(line: 10, column: 3, scope: !10)
!12 = distinct !DISubprogram(name: "loop_contains_store_multidim_condition_load", scope: !2, file: !2, line: 20, type: !3, isLocal: false, isDefinition: true, scopeLine: 20, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!13 = !DILocation(line: 20, column: 3, scope: !12)
!16 = distinct !DISubprogram(name: "novec_loop_contains_store_ee_condition_is_invariant", scope: !2, file: !2, line: 40, type: !3, isLocal: false, isDefinition: true, scopeLine: 40, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!17 = !DILocation(line: 40, column: 3, scope: !16)
!18 = distinct !DISubprogram(name: "loop_contains_store_fcmp_condition", scope: !2, file: !2, line: 50, type: !3, isLocal: false, isDefinition: true, scopeLine: 50, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!19 = !DILocation(line: 50, column: 3, scope: !18)
!20 = distinct !DISubprogram(name: "loop_contains_store_safe_dependency", scope: !2, file: !2, line: 60, type: !3, isLocal: false, isDefinition: true, scopeLine: 60, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!21 = !DILocation(line: 60, column: 3, scope: !20)
!22 = distinct !DISubprogram(name: "loop_contains_store_unsafe_dependency", scope: !2, file: !2, line: 70, type: !3, isLocal: false, isDefinition: true, scopeLine: 70, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!23 = !DILocation(line: 70, column: 3, scope: !22)
!30 = distinct !DISubprogram(name: "novec_loop_contains_store_volatile", scope: !2, file: !2, line: 110, type: !3, isLocal: false, isDefinition: true, scopeLine: 110, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!31 = !DILocation(line: 110, column: 3, scope: !30)
!32 = distinct !DISubprogram(name: "loop_contains_store_to_invariant_location", scope: !2, file: !2, line: 120, type: !3, isLocal: false, isDefinition: true, scopeLine: 120, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!33 = !DILocation(line: 120, column: 3, scope: !32)
!36 = distinct !DISubprogram(name: "loop_contains_store_requiring_alias_check", scope: !2, file: !2, line: 140, type: !3, isLocal: false, isDefinition: true, scopeLine: 140, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!37 = !DILocation(line: 140, column: 3, scope: !36)
!38 = distinct !DISubprogram(name: "loop_contains_store_decrementing_iv", scope: !2, file: !2, line: 150, type: !3, isLocal: false, isDefinition: true, scopeLine: 150, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!39 = !DILocation(line: 150, column: 3, scope: !38)
!40 = distinct !DISubprogram(name: "loop_contains_store_condition_load_requires_gather", scope: !2, file: !2, line: 160, type: !3, isLocal: false, isDefinition: true, scopeLine: 160, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!41 = !DILocation(line: 160, column: 3, scope: !40)
!42 = distinct !DISubprogram(name: "loop_contains_store_uncounted_exit_is_a_switch", scope: !2, file: !2, line: 170, type: !3, isLocal: false, isDefinition: true, scopeLine: 170, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!43 = !DILocation(line: 170, column: 3, scope: !42)
!44 = distinct !DISubprogram(name: "loop_contains_store_uncounted_exit_is_not_guaranteed_to_execute", scope: !2, file: !2, line: 180, type: !3, isLocal: false, isDefinition: true, scopeLine: 180, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!45 = !DILocation(line: 180, column: 3, scope: !44)
!46 = distinct !DISubprogram(name: "test_nodep", scope: !2, file: !2, line: 190, type: !3, isLocal: false, isDefinition: true, scopeLine: 190, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!47 = !DILocation(line: 190, column: 3, scope: !46)
!48 = distinct !DISubprogram(name: "histogram_with_uncountable_exit", scope: !2, file: !2, line: 200, type: !3, isLocal: false, isDefinition: true, scopeLine: 200, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!49 = !DILocation(line: 200, column: 3, scope: !48)
!50 = distinct !DISubprogram(name: "loop_contains_store_between_two_early_exits", scope: !2, file: !2, line: 210, type: !3, isLocal: false, isDefinition: true, scopeLine: 210, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!51 = !DILocation(line: 210, column: 3, scope: !50)
!52 = distinct !DISubprogram(name: "loop_contains_store_before_two_early_exits", scope: !2, file: !2, line: 220, type: !3, isLocal: false, isDefinition: true, scopeLine: 220, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!53 = !DILocation(line: 220, column: 3, scope: !52)
!54 = distinct !DISubprogram(name: "one_uncountable_two_countable_exits", scope: !2, file: !2, line: 230, type: !3, isLocal: false, isDefinition: true, scopeLine: 230, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!55 = !DILocation(line: 230, column: 3, scope: !54)
!56 = distinct !DISubprogram(name: "uncountable_exit_with_reduction", scope: !2, file: !2, line: 240, type: !3, isLocal: false, isDefinition: true, scopeLine: 240, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!57 = !DILocation(line: 240, column: 3, scope: !56)
!60 = distinct !DISubprogram(name: "uncountable_exit_with_constant_nonunit_stride", scope: !2, file: !2, line: 260, type: !3, isLocal: false, isDefinition: true, scopeLine: 260, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!61 = !DILocation(line: 260, column: 3, scope: !60)
!62 = distinct !DISubprogram(name: "uncountable_exit_with_invariant_but_unknown_stride", scope: !2, file: !2, line: 270, type: !3, isLocal: false, isDefinition: true, scopeLine: 270, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!63 = !DILocation(line: 270, column: 3, scope: !62)
!64 = distinct !DISubprogram(name: "uncountable_exit_condition_load_offset_from_iv", scope: !2, file: !2, line: 280, type: !3, isLocal: false, isDefinition: true, scopeLine: 280, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!65 = !DILocation(line: 280, column: 3, scope: !64)
!66 = distinct !DISubprogram(name: "uncountable_exit_with_masked_ldst_separate_condition", scope: !2, file: !2, line: 290, type: !3, isLocal: false, isDefinition: true, scopeLine: 290, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!67 = !DILocation(line: 290, column: 3, scope: !66)
!68 = distinct !DISubprogram(name: "novec_uncountable_exit_condition_address_is_invariant", scope: !2, file: !2, line: 300, type: !3, isLocal: false, isDefinition: true, scopeLine: 300, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!69 = !DILocation(line: 300, column: 3, scope: !68)
!70 = distinct !DISubprogram(name: "novec_uncountable_exit_condition_address_is_addrec_in_outer_loop", scope: !2, file: !2, line: 310, type: !3, isLocal: false, isDefinition: true, scopeLine: 310, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!71 = !DILocation(line: 310, column: 3, scope: !70)
!72 = distinct !DISubprogram(name: "crash_conditional_load_for_uncountable_exit", scope: !2, file: !2, line: 320, type: !3, isLocal: false, isDefinition: true, scopeLine: 320, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!73 = !DILocation(line: 320, column: 3, scope: !72)
!74 = distinct !DISubprogram(name: "crash_conditional_load_for_uncountable_exit_argptr", scope: !2, file: !2, line: 330, type: !3, isLocal: false, isDefinition: true, scopeLine: 330, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!75 = !DILocation(line: 330, column: 3, scope: !74)
!76 = distinct !DISubprogram(name: "combined_exit_conditions", scope: !2, file: !2, line: 340, type: !3, isLocal: false, isDefinition: true, scopeLine: 340, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!77 = !DILocation(line: 340, column: 3, scope: !76)
!78 = distinct !DISubprogram(name: "uncountable_exit_with_extra_induction", scope: !2, file: !2, line: 350, type: !3, isLocal: false, isDefinition: true, scopeLine: 350, flags: DIFlagPrototyped, isOptimized: true, unit: !1000, retainedNodes: !4)
!79 = !DILocation(line: 350, column: 3, scope: !78)
!1000 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang", file: !2, isOptimized: true, flags: "-O2", splitDebugFilename: "abc.debug", emissionKind: 2)
