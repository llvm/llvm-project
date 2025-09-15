; RUN: opt -passes=tsan -tsan-use-dominance-analysis < %s -S | FileCheck %s

; This file contains tests for the TSan dominance-based optimization.
; We check that redundant instrumentation is removed when one access
; dominates/post-dominates another, and is NOT removed when the path between
; them is "dirty".

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; --- Global variables for testing (minimized to two) ---
@g1 = common global i32 0, align 4
@g2 = common global i32 0, align 4

; --- External Function Declarations for Tests ---
declare void @some_external_call()
declare void @safe_func() #0
declare void @external_check()

; =============================================================================
; INTRA-BLOCK DOMINANCE TESTS
; =============================================================================

define void @test_intra_block_write_write() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_intra_block_write_write
; CHECK:      call void @__tsan_write4(ptr @g1)
; The second write is dominated and should NOT be instrumented.
; CHECK-NOT:  call void @__tsan_write4(ptr @g1)
; CHECK:      ret void

define void @test_intra_block_write_read() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  %val = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_intra_block_write_read
; CHECK:      call void @__tsan_write4(ptr @g1)
; The read is dominated and should NOT be instrumented.
; CHECK-NOT:  call void @__tsan_read4(ptr @g1)
; CHECK:      ret void

define void @test_intra_block_read_read() nounwind uwtable sanitize_thread {
entry:
  %val1 = load i32, ptr @g1, align 4
  %val2 = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_intra_block_read_read
; CHECK:      call void @__tsan_read4(ptr @g1)
; The second read is dominated and should NOT be instrumented.
; CHECK-NOT:  call void @__tsan_read4(ptr @g1)
; CHECK:      ret void

; =============================================================================
; PATH CLEARNESS TESTS
; =============================================================================

define void @test_path_not_clear_call() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  call void @some_external_call()
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_path_not_clear_call
; CHECK:      call void @__tsan_write4(ptr @g1)
; An unsafe call makes the path dirty. Optimization must NOT trigger.
; CHECK:      call void @__tsan_write4(ptr @g1)
; CHECK:      ret void

define void @test_path_clear_safe_call() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  call void @safe_func()
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_path_clear_safe_call
; CHECK:      call void @__tsan_write4(ptr @g1)
; A safe intrinsic call should not block the optimization.
; CHECK-NOT:  call void @__tsan_write4(ptr @g1)
; CHECK:      ret void

; =============================================================================
; INTER-BLOCK DOMINANCE TESTS
; =============================================================================

define void @test_inter_block_dom(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br i1 %cond, label %if.then, label %if.else
if.then:
  store i32 2, ptr @g1, align 4
  br label %if.end
if.else:
  store i32 3, ptr @g1, align 4
  br label %if.end
if.end:
  ret void
}
; CHECK-LABEL: define void @test_inter_block_dom
; CHECK:      call void @__tsan_write4(ptr @g1)
; CHECK:      if.then:
; CHECK-NOT:  call void @__tsan_write4(ptr @g1)
; CHECK:      if.else:
; CHECK-NOT:  call void @__tsan_write4(ptr @g1)
; CHECK:      ret void

; =============================================================================
; POST-DOMINANCE TESTS
; =============================================================================

define void @test_post_dom(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  store i32 2, ptr @g1, align 4
  br label %if.end
if.else:
  store i32 3, ptr @g1, align 4
  br label %if.end
if.end:
  store i32 4, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_post_dom
; CHECK:      if.then:
; CHECK-NOT:  call void @__tsan_write4(ptr @g1)
; CHECK:      if.else:
; CHECK-NOT:  call void @__tsan_write4(ptr @g1)
; CHECK:      if.end:
; CHECK:      call void @__tsan_write4(ptr @g1)
; CHECK:      ret void

; =============================================================================
; ALIAS ANALYSIS TESTS
; =============================================================================

; Simple alias analysis: no alias.
define void @test_no_alias() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  store i32 2, ptr @g2, align 4
  ret void
}
; CHECK-LABEL: define void @test_no_alias
; CHECK:      call void @__tsan_write4(ptr @g1)
; Different addresses. The optimization must NOT trigger.
; CHECK:      call void @__tsan_write4(ptr @g2)
; CHECK:      ret void

; MustAlias via zero-index GEP (should eliminate)
define void @alias_mustalias_gep0() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  %p = getelementptr i32, ptr @g1, i64 0
  %v = load i32, ptr %p, align 4
  ret void
}
; CHECK-LABEL: define void @alias_mustalias_gep0
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK-NOT:   call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; Different offsets within the same base object (should NOT eliminate)
@arr = common global [5 x i32] zeroinitializer, align 4
define void @alias_different_offsets() nounwind uwtable sanitize_thread {
entry:
  %p0 = getelementptr [5 x i32], ptr @arr, i64 0, i64 0
  %p1 = getelementptr [5 x i32], ptr @arr, i64 0, i64 1
  store i32 1, ptr %p0, align 4
  store i32 2, ptr %p1, align 4
  ret void
}
; CHECK-LABEL: define void @alias_different_offsets
; CHECK:       call void @__tsan_write4(
; CHECK:       call void @__tsan_write4(
; CHECK:       ret void

; Equal offsets within the same base object (should eliminate)
define void @alias_same_offsets() nounwind uwtable sanitize_thread {
entry:
  %p0 = getelementptr [5 x i32], ptr @arr, i64 0, i64 1
  %p1 = getelementptr [5 x i32], ptr @arr, i64 0, i64 1
  store i32 1, ptr %p0, align 4
  store i32 2, ptr %p1, align 4
  ret void
}
; CHECK-LABEL: define void @alias_same_offsets
; CHECK:       call void @__tsan_write4(
; CHECK-NOT:   call void @__tsan_write4(
; CHECK:       ret void


; MayAlias via phi of two globals (should NOT eliminate)
define void @alias_mayalias_phi(i1 %c) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br i1 %c, label %A, label %B
A:
  br label %join
B:
  br label %join
join:
  %p = phi ptr [ @g1, %A ], [ @g2, %B ]
  %v = load i32, ptr %p, align 4
  ret void
}
; CHECK-LABEL: define void @alias_mayalias_phi
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       join:
; CHECK:       call void @__tsan_read4(
; CHECK:       ret void

; Pointer round-trip via ptrtoint/inttoptr (typically breaks MustAlias)
; (should NOT eliminate)
define void @alias_ptr_roundtrip() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  %i = ptrtoint ptr @g1 to i64
  %p2 = inttoptr i64 %i to ptr
  %v = load i32, ptr %p2, align 4
  ret void
}
; CHECK-LABEL: define void @alias_ptr_roundtrip
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       call void @__tsan_read4(
; CHECK:       ret void

; Bitcast-based MustAlias (i32* <-> i8*) (should eliminate)
define void @alias_bitcast_i8_i32() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  %p8 = bitcast ptr @g1 to ptr
  %v = load i32, ptr %p8, align 4
  ret void
}
; CHECK-LABEL: define void @alias_bitcast_i8_i32
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK-NOT:   call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; GEP with folded zero offset is MustAlias: (%n - %n) -> 0
define void @alias_gep_folded_zero(i64 %n) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  %t = sub i64 %n, %n
  %p = getelementptr i32, ptr @g1, i64 %t
  %v = load i32, ptr %p, align 4
  ret void
}
; CHECK-LABEL: define void @alias_gep_folded_zero
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK-NOT:   call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

define void @alias_select_same_ptr(i1 %c) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  %p = select i1 %c, ptr @g1, ptr @g1
  %v = load i32, ptr %p, align 4
  ret void
}
; CHECK-LABEL: define void @alias_select_same_ptr
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK-NOT:   call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; =============================================================================
; BRANCHING WITH MULTIPLE PATHS (one path dirty)
; =============================================================================

; Case A: inter-BB with a diamond where one branch is dirty.
; Path entry -> then (unsafe) -> merge, and entry -> else (safe) -> merge.
define void @multi_path_inter_dirty(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br i1 %cond, label %then, label %else

then:
  call void @some_external_call()
  br label %merge

else:
  call void @safe_func()
  br label %merge

merge:
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @multi_path_inter_dirty
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; Dirty along one path => must instrument at merge.
; CHECK:       merge:
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; Case B: inter-BB where both branches are safe (no dangerous instr). Should eliminate.
define void @multi_path_inter_clean(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br i1 %cond, label %then, label %else

then:
  call void @safe_func()
  br label %merge

else:
  call void @safe_func()
  br label %merge

merge:
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @multi_path_inter_clean
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; Both paths clean => dominated read at merge should be removed.
; CHECK:       merge:
; CHECK-NOT:   call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; =============================================================================
; MIXED: intra-BB safe suffix vs. inter-BB dirty path
; =============================================================================
define void @mixed_intra_inter(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  ; intra-BB suffix between store and next store is safe (no calls)
  store i32 1, ptr @g1, align 4
  store i32 2, ptr @g1, align 4
  br i1 %cond, label %dirty, label %clean

dirty:
  ; dangerous call on one path
  call void @some_external_call()
  br label %merge

clean:
  ; safe on other path
  call void @safe_func()
  br label %merge

merge:
  ; must keep because one incoming path is dirty
  store i32 3, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @mixed_intra_inter
; First store instruments.
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; Second store in same BB is dominated by the first and safe => removed.
; CHECK-NOT:   call void @__tsan_write4(ptr @g1)
; Final store must remain due to dirty path.
; CHECK:       merge:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; =============================================================================
; POST-DOM with dirty suffix at start BB blocks elimination (renamed BBs)
; =============================================================================
define void @postdom_dirty_start_suffix(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  ; Initial write
  store i32 1, ptr @g1, align 4
  ; Dirty suffix in the start block blocks elimination
  call void @some_external_call()
  br i1 %cond, label %path_then, label %path_else

path_then:
  br label %merge

path_else:
  br label %merge

merge:
  ; Despite post-dominance, path is not clear due to dirty suffix in entry
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @postdom_dirty_start_suffix
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       call void @some_external_call()
; CHECK:       merge:
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; =============================================================================
; DIRTY PREFIX IN END BB blocks elimination (prefixSafe)
; =============================================================================
define void @dirty_prefix_in_end_bb() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br label %end

end:
  ; Dirty prefix in the end block before the target access
  call void @some_external_call()
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @dirty_prefix_in_end_bb
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       end:
; CHECK:       call void @some_external_call()
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; =============================================================================
; IRRELEVANT DIRTY PATH NOT REACHING EndBB should not block elimination
; =============================================================================
define void @dirty_unrelated_cone(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br i1 %cond, label %to_end, label %to_dead

to_end:
  br label %end

to_dead:
  ; Dirty path that does NOT reach %end at all
  call void @some_external_call()
  br label %dead

dead:
  ret void

end:
  %v = load i32, ptr @g1, align 4
  ret void
}
; The dirty path is outside the cone to %end, so read can be eliminated.
; CHECK-LABEL: define void @dirty_unrelated_cone
; CHECK:       entry:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       end:
; CHECK-NOT:   call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; =============================================================================
; POST-DOMINANCE WITH LOOP
; =============================================================================
define void @postdom_loop() nounwind uwtable sanitize_thread {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %call = call i32 (...) @external_check()
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  store i32 1, ptr @g1, align 4
  br label %while.cond

while.end:                                        ; preds = %while.cond
  store i32 2, ptr @g1, align 4
  ret void
}
; It's a potentially infinite loop,
; so the store in while.end should not be eliminated.
; CHECK-LABEL: define void @postdom_loop
; CHECK:       while.body:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       while.end:
; CHECK:       call void @__tsan_write4(ptr @g1)

; =============================================================================
; POST-DOMINANCE BLOCKED BY POTENTIAL INFINITE LOOP (CallBase check)
; =============================================================================

declare void @dom_safe_but_postdom_unsafe() #0

define void @test_post_dom_blocked_by_readnone_call(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  ; dominated by neither entry nor if.end (in terms of domination elimination flow)
  ; but post-dominated by if.end
  store i32 1, ptr @g1, align 4
  call void @dom_safe_but_postdom_unsafe()
  br label %if.end
if.else:
  br label %if.end
if.end:
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_post_dom_blocked_by_readnone_call
; CHECK:       if.then:
; The call is readnone (safe for sync) but not an intrinsic (unsafe for termination).
; So the first write must NOT be eliminated (post-dominance is blocked).
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       call void @dom_safe_but_postdom_unsafe()
; CHECK:       if.end:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

declare void @postdom_safe_func() #1

define void @test_post_dom_allowed_by_postdome_safe_func(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  store i32 1, ptr @g1, align 4
  call void @postdom_safe_func()
  br label %if.end
if.else:
  br label %if.end
if.end:
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @test_post_dom_allowed_by_postdome_safe_func
; CHECK:       if.then:
; CHECK-NOT:   call void @__tsan_write4(ptr @g1)
; CHECK:       call void @postdom_safe_func()
; CHECK:       if.end:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; Attributes for the "safe" function
attributes #0 = { nosync }
attributes #1 = { nosync willreturn nounwind }
