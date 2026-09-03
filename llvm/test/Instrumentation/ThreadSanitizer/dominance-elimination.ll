; RUN: opt < %s -passes=tsan -S | FileCheck %s
; RUN: opt < %s -passes=tsan -tsan-use-dominance-analysis=false -S | FileCheck %s --check-prefix=NODOM
; RUN: opt < %s -passes=tsan -tsan-distinguish-volatile -S | FileCheck %s --check-prefix=VOLATILE

; Tests for TSan dominance-based redundant instrumentation elimination.
; Redundant instrumentation is removed when one access dominates another to
; the same location with no synchronization on any path between them.
;
; Check prefixes:
;   CHECK   - default run (optimization enabled)
;   NODOM   - optimization disabled; all accesses must remain instrumented
;   VOLATILE - -tsan-distinguish-volatile enabled; volatile and non-volatile
;              accesses emit different runtime calls and must not be merged

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@g1 = global i32 0, align 4
@g2 = global i32 0, align 4
@arr = global [5 x i32] zeroinitializer, align 4

; Unsafe call (no nosync): blocks dominance-based elimination.
declare void @ext_call()
; nosync call: safe to cross for dominance-based elimination.
declare void @nosync_func() #0
; Unsafe function returning i32, used in loop tests.
declare i32 @ext_check(...)

; ===========================================================================
; Intra-block dominance
; ===========================================================================

; First write dominates second write to the same location.
define void @intra_block_write_write() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @intra_block_write_write
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK-NOT:   call void @__tsan_write4(ptr @g1)
; CHECK:       ret void
;
; NODOM-LABEL: define void @intra_block_write_write
; NODOM:       call void @__tsan_write4(ptr @g1)
; NODOM:       call void @__tsan_write4(ptr @g1)
; NODOM:       ret void

; Write dominates following read: write covers read, so read is removed.
define void @intra_block_write_read() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  %val = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @intra_block_write_read
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK-NOT:   call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; First read dominates second read to the same location.
define void @intra_block_read_read() nounwind uwtable sanitize_thread {
entry:
  %v1 = load i32, ptr @g1, align 4
  %v2 = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @intra_block_read_read
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK-NOT:   call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; A dominating read does NOT eliminate a write on a dominated path.
; The read-before-write elimination in chooseInstructionsToInstrument only
; applies within the same basic block, so this uses an inter-block scenario.
; The write is only on one branch so it is NOT the post-dominator of the
; read; the dominance check therefore applies in isolation.
define void @dom_read_does_not_cover_write(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  %v = load i32, ptr @g1, align 4
  br i1 %cond, label %write.path, label %skip
write.path:
  store i32 1, ptr @g1, align 4
  br label %end
skip:
  br label %end
end:
  ret void
}
; CHECK-LABEL: define void @dom_read_does_not_cover_write
; The read dominates the write, but a read cannot cover write-write races.
; Both must remain instrumented.
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       write.path:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; ===========================================================================
; Path safety
; ===========================================================================

; An inter-thread atomic on the path is a synchronization point: no elimination.
; (isTsanAtomic returns true for cross-thread scope.)
define void @path_dirty_atomic_interthread(ptr %p, ptr %flag) nounwind sanitize_thread {
entry:
  store i32 1, ptr %p, align 4
  %v = load atomic i32, ptr %flag acquire, align 4
  store i32 2, ptr %p, align 4
  ret void
}
; CHECK-LABEL: define void @path_dirty_atomic_interthread
; CHECK:       call void @__tsan_write4(ptr %p)
; CHECK:       call void @__tsan_write4(ptr %p)
; CHECK:       ret void

; A singlethread-scoped atomic on the path does NOT block elimination.
; isTsanAtomic returns false for singlethread scope on load/store, so
; it is treated as a plain access with no inter-thread synchronization.
define void @path_clear_atomic_singlethread(ptr %p, ptr %q) nounwind sanitize_thread {
entry:
  store i32 1, ptr %p, align 4
  store atomic i32 0, ptr %q syncscope("singlethread") seq_cst, align 4
  store i32 2, ptr %p, align 4
  ret void
}
; CHECK-LABEL: define void @path_clear_atomic_singlethread
; CHECK:       call void @__tsan_write4(ptr %p)
; CHECK-NOT:   call void @__tsan_write4(ptr %p)
; CHECK:       ret void

; A singlethread fence IS treated as unsafe (conservative: isTsanAtomic
; returns true for any non-load/store instruction with a sync scope,
; regardless of whether the scope is singlethread or system). Both writes
; remain instrumented.
define void @path_dirty_fence_singlethread(ptr %p) nounwind sanitize_thread {
entry:
  store i32 1, ptr %p, align 4
  fence syncscope("singlethread") seq_cst
  store i32 2, ptr %p, align 4
  ret void
}
; CHECK-LABEL: define void @path_dirty_fence_singlethread
; CHECK:       call void @__tsan_write4(ptr %p)
; CHECK:       call void @__tsan_write4(ptr %p)
; CHECK:       ret void

; An unsafe call between two accesses makes the path dirty: no elimination.
define void @path_dirty_call() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  call void @ext_call()
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @path_dirty_call
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; A nosync call does not block dominance elimination.
define void @path_clear_nosync_call() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  call void @nosync_func()
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @path_clear_nosync_call
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK-NOT:   call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; An indirect call with nosync at the call site does not block elimination.
; #0 = { nosync } — the attribute group is defined at the bottom of this file.
; This exercises the call-site attribute path in isInstrSafe (hasFnAttr).
define void @path_clear_indirect_nosync_callsite(ptr %fn) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  call void %fn() #0
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @path_clear_indirect_nosync_callsite
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK-NOT:   call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; ===========================================================================
; Inter-block dominance
; ===========================================================================

; A write in the entry block dominates writes in both branches of a diamond.
define void @inter_block_dom(i1 %cond) nounwind uwtable sanitize_thread {
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
; CHECK-LABEL: define void @inter_block_dom
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       if.then:
; CHECK-NOT:   call void @__tsan_write4(ptr @g1)
; CHECK:       if.else:
; CHECK-NOT:   call void @__tsan_write4(ptr @g1)
; CHECK:       ret void
;
; NODOM-LABEL: define void @inter_block_dom
; NODOM:       call void @__tsan_write4(ptr @g1)
; NODOM:       call void @__tsan_write4(ptr @g1)
; NODOM:       call void @__tsan_write4(ptr @g1)
; NODOM:       ret void

; ===========================================================================
; Diamonds: an access at the merge point has no dominating same-location access
;
; Only dominance is exploited here, so an access whose only "relative" is in a
; sibling branch or at the merge point (which post-dominates, but does not
; dominate, the branch accesses) is never eliminated. These confirm the pass
; does not over-eliminate in such shapes.
; ===========================================================================

; Writes in both branches and at the merge: none dominates another, since the
; entry block has no write to @g1. All three remain.
define void @diamond_branch_and_merge_writes(i1 %cond) nounwind uwtable sanitize_thread {
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
; CHECK-LABEL: define void @diamond_branch_and_merge_writes
; CHECK:       if.then:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       if.else:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       if.end:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void
;
; NODOM-LABEL: define void @diamond_branch_and_merge_writes
; NODOM:       call void @__tsan_write4(ptr @g1)
; NODOM:       call void @__tsan_write4(ptr @g1)
; NODOM:       call void @__tsan_write4(ptr @g1)
; NODOM:       ret void

; A write at the merge does not dominate the branch reads (entry has no access),
; so the reads are not eliminated. All three accesses remain.
define void @diamond_merge_write_branch_reads(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  %v1 = load i32, ptr @g1, align 4
  br label %if.end
if.else:
  %v2 = load i32, ptr @g1, align 4
  br label %if.end
if.end:
  store i32 0, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @diamond_merge_write_branch_reads
; CHECK:       if.then:
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       if.else:
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       if.end:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; A read at the merge does not dominate the branch reads. All three remain.
define void @diamond_merge_read_branch_reads(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  %v1 = load i32, ptr @g1, align 4
  br label %if.end
if.else:
  %v2 = load i32, ptr @g1, align 4
  br label %if.end
if.end:
  %v3 = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @diamond_merge_read_branch_reads
; CHECK:       if.then:
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       if.else:
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       if.end:
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; Writes in both branches with a read at the merge: the merge read does not
; dominate the branch writes (and a read could not cover a write anyway). Both
; writes remain instrumented.
define void @diamond_merge_read_branch_writes(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  store i32 1, ptr @g1, align 4
  br label %if.end
if.else:
  store i32 2, ptr @g1, align 4
  br label %if.end
if.end:
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @diamond_merge_read_branch_writes
; CHECK:       if.then:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       if.else:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       if.end:
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; ===========================================================================
; Multi-path: dirty vs clean
; ===========================================================================

; One branch carries an unsafe call: the read at the merge is not eliminated.
define void @multi_path_dirty(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br i1 %cond, label %then, label %else
then:
  call void @ext_call()
  br label %merge
else:
  call void @nosync_func()
  br label %merge
merge:
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @multi_path_dirty
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       merge:
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; Both branches carry only nosync calls: the read at the merge is eliminated.
define void @multi_path_clean(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br i1 %cond, label %then, label %else
then:
  call void @nosync_func()
  br label %merge
else:
  call void @nosync_func()
  br label %merge
merge:
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @multi_path_clean
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       merge:
; CHECK-NOT:   call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; Adjacent stores in the same block are eliminated; a dirty path to the merge
; block keeps the final store instrumented.
define void @mixed_intra_inter(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  store i32 2, ptr @g1, align 4
  br i1 %cond, label %dirty, label %clean
dirty:
  call void @ext_call()
  br label %merge
clean:
  call void @nosync_func()
  br label %merge
merge:
  store i32 3, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @mixed_intra_inter
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK-NOT:   call void @__tsan_write4(ptr @g1)
; CHECK:       merge:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; ===========================================================================
; Path safety: dirty suffix / prefix / unrelated path
; ===========================================================================

; Unsafe call in the suffix of the dominating block (Region 1) blocks
; elimination of the dominated read at the merge.
define void @dom_dirty_dom_suffix(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  call void @ext_call()
  br i1 %cond, label %path.then, label %path.else
path.then:
  br label %merge
path.else:
  br label %merge
merge:
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @dom_dirty_dom_suffix
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       merge:
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; Unsafe call in the prefix of the dominated block (Region 2) blocks
; elimination.
define void @dom_dirty_curr_prefix() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br label %end
end:
  call void @ext_call()
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @dom_dirty_curr_prefix
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; A dirty intermediate path that does not reach the dominated block must not
; block elimination (Region 3 restricts the scan to the reverse-reachable cone).
define void @dom_dirty_unrelated_path(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  br i1 %cond, label %to.end, label %to.dead
to.end:
  br label %end
to.dead:
  call void @ext_call()
  br label %dead
dead:
  ret void
end:
  %v = load i32, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @dom_dirty_unrelated_path
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       end:
; CHECK-NOT:   call void @__tsan_read4(ptr @g1)
; CHECK:       ret void

; ===========================================================================
; Loops: a store in the body and a store after the loop, with no dominance
;
; The loop may execute zero times (entry -> cond -> end), so the body store does
; not dominate the exit store; neither store dominates the other and both remain.
; ===========================================================================

; The loop condition also contains an unsafe call.
define void @loop_body_and_exit_writes_unsafe_cond() nounwind uwtable sanitize_thread {
entry:
  br label %while.cond
while.cond:
  %v = call i32 (...) @ext_check()
  %tobool = icmp ne i32 %v, 0
  br i1 %tobool, label %while.body, label %while.end
while.body:
  store i32 1, ptr @g1, align 4
  br label %while.cond
while.end:
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @loop_body_and_exit_writes_unsafe_cond
; CHECK:       while.body:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       while.end:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; Pure loop (only branch instructions, no calls or atomics).
define void @loop_body_and_exit_writes_pure(i1 %cond) nounwind uwtable sanitize_thread {
entry:
  br label %while.cond
while.cond:
  br i1 %cond, label %while.body, label %while.end
while.body:
  store i32 1, ptr @g1, align 4
  br label %while.cond
while.end:
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @loop_body_and_exit_writes_pure
; CHECK:       while.body:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       while.end:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; ===========================================================================
; Loops: a dominator OUTSIDE the loop covering an access INSIDE the loop
;
; When the dominating access A is outside the loop, it executes only once, so
; it can only cover the loop access B if the back-edge path B -> ... -> B is
; also synchronization-free.  A synchronization on that back-edge path means
; later iterations of B are no longer covered by A and must stay instrumented.
; ===========================================================================

; Clean loop body: A in the entry block dominates B in the loop, and the whole
; loop (including the back-edge) is sync-free, so B is eliminated.
define void @loop_dom_clean(i1 %c) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4          ; A
  br label %loop
loop:
  store i32 2, ptr @g1, align 4          ; B
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
; CHECK-LABEL: define void @loop_dom_clean
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       loop:
; CHECK-NOT:   call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; A nosync call on the back-edge is still safe: B is eliminated.  This confirms
; the loop handling does not over-conservatively reject sync-free loops.
define void @loop_dom_clean_nosync(i1 %c) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4          ; A
  br label %loop
loop:
  store i32 2, ptr @g1, align 4          ; B
  call void @nosync_func()
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
; CHECK-LABEL: define void @loop_dom_clean_nosync
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       loop:
; CHECK-NOT:   call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; An inter-thread acquire AFTER B in the loop block sits on the back-edge path
; between consecutive executions of B.  A (entry) cannot cover those later
; executions: both stores must remain instrumented.
define void @loop_dom_dirty_atomic_tail(i1 %c) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4          ; A
  br label %loop
loop:
  store i32 2, ptr @g1, align 4          ; B
  %v = load atomic i32, ptr @g2 acquire, align 4
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
; CHECK-LABEL: define void @loop_dom_dirty_atomic_tail
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       loop:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; Same hazard, but the synchronization is in a separate latch (back-edge) block
; rather than in B's own block.  Both stores must remain instrumented.
define void @loop_dom_dirty_atomic_latch(i1 %c) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4          ; A
  br label %header
header:
  store i32 2, ptr @g1, align 4          ; B
  br label %latch
latch:
  %v = load atomic i32, ptr @g2 acquire, align 4
  br i1 %c, label %header, label %exit
exit:
  ret void
}
; CHECK-LABEL: define void @loop_dom_dirty_atomic_latch
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       header:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; An unsafe (non-nosync) call on the back-edge also blocks elimination.
define void @loop_dom_dirty_call_tail(i1 %c) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4          ; A
  br label %loop
loop:
  store i32 2, ptr @g1, align 4          ; B
  call void @ext_call()
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
; CHECK-LABEL: define void @loop_dom_dirty_call_tail
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       loop:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; Nested loops with the synchronization in the OUTER latch: B in the inner loop
; re-executes across outer iterations with the sync in between, and the
; back-edge scan must reach the outer latch through the cone. Both stores remain.
define void @loop_dom_nested_outer_sync(i1 %c1, i1 %c2) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4          ; A
  br label %outer
outer:
  br label %inner
inner:
  store i32 2, ptr @g1, align 4          ; B
  br i1 %c2, label %inner, label %outer.latch
outer.latch:
  %v = load atomic i32, ptr @g2 acquire, align 4
  br i1 %c1, label %outer, label %exit
exit:
  ret void
}
; CHECK-LABEL: define void @loop_dom_nested_outer_sync
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       inner:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; A diamond on the back-edge path with synchronization on one of its branches:
; the back-edge cone scan must visit both branches. Both stores remain.
define void @loop_dom_diamond_backedge_sync(i1 %c, i1 %d) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4          ; A
  br label %loop
loop:
  store i32 2, ptr @g1, align 4          ; B
  br i1 %d, label %ba, label %bb
ba:
  %v = load atomic i32, ptr @g2 acquire, align 4
  br label %latch
bb:
  br label %latch
latch:
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
; CHECK-LABEL: define void @loop_dom_diamond_backedge_sync
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       loop:
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; Nested loops, fully sync-free: B is still eliminated (precision preserved
; through the back-edge cone scan).
define void @loop_dom_nested_clean(i1 %c1, i1 %c2) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4          ; A
  br label %outer
outer:
  br label %inner
inner:
  store i32 2, ptr @g1, align 4          ; B
  br i1 %c2, label %inner, label %outer.latch
outer.latch:
  br i1 %c1, label %outer, label %exit
exit:
  ret void
}
; CHECK-LABEL: define void @loop_dom_nested_clean
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       inner:
; CHECK-NOT:   call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; ===========================================================================
; Synchronization barriers
; ===========================================================================

; An atomic RMW operation is a synchronization point: no elimination.
define void @atomic_blocks_dom_elim() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  %old = atomicrmw add ptr @g1, i32 1 seq_cst
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @atomic_blocks_dom_elim
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; A seq_cst fence is a synchronization point: no elimination.
define void @fence_blocks_dom_elim() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  fence seq_cst
  store i32 2, ptr @g1, align 4
  ret void
}
; CHECK-LABEL: define void @fence_blocks_dom_elim
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       ret void

; ===========================================================================
; Access size compatibility
; ===========================================================================

; A dominating write of smaller size must NOT eliminate a larger write: the
; 4-byte instrumentation does not cover bytes 4-7 of the 8-byte access.
define void @size_mismatch_small_dom(ptr %p) nounwind sanitize_thread {
entry:
  store i32 1, ptr %p, align 4
  store i64 2, ptr %p, align 8
  ret void
}
; CHECK-LABEL: define void @size_mismatch_small_dom
; CHECK:       call void @__tsan_write4(ptr %p)
; CHECK:       call void @__tsan_write8(ptr %p)
; CHECK:       ret void

; A dominating write of larger size CAN eliminate a smaller write: write8
; covers the full [p, p+4) range that write4 would instrument.
define void @size_mismatch_large_dom(ptr %p) nounwind sanitize_thread {
entry:
  store i64 1, ptr %p, align 8
  store i32 2, ptr %p, align 4
  ret void
}
; CHECK-LABEL: define void @size_mismatch_large_dom
; CHECK:       call void @__tsan_write8(ptr %p)
; CHECK-NOT:   call void @__tsan_write4(ptr %p)
; CHECK:       ret void

; ===========================================================================
; Alias analysis
; ===========================================================================

; Accesses to distinct globals are NoAlias: no elimination.
define void @no_alias() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  store i32 2, ptr @g2, align 4
  ret void
}
; CHECK-LABEL: define void @no_alias
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       call void @__tsan_write4(ptr @g2)
; CHECK:       ret void

; Zero-index GEP is MustAlias with the base: elimination fires.
define void @mustalias_gep0() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  %p = getelementptr i32, ptr @g1, i64 0
  %v = load i32, ptr %p, align 4
  ret void
}
; CHECK-LABEL: define void @mustalias_gep0
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK-NOT:   call void @__tsan_read4(
; CHECK:       ret void

; Different GEP offsets into the same array are NoAlias: no elimination.
define void @noalias_different_offsets() nounwind uwtable sanitize_thread {
entry:
  %p0 = getelementptr [5 x i32], ptr @arr, i64 0, i64 0
  %p1 = getelementptr [5 x i32], ptr @arr, i64 0, i64 1
  store i32 1, ptr %p0, align 4
  store i32 2, ptr %p1, align 4
  ret void
}
; CHECK-LABEL: define void @noalias_different_offsets
; CHECK:       call void @__tsan_write4(
; CHECK:       call void @__tsan_write4(
; CHECK:       ret void

; Identical GEP offsets into the same array are MustAlias: elimination fires.
define void @mustalias_same_offsets() nounwind uwtable sanitize_thread {
entry:
  %p0 = getelementptr [5 x i32], ptr @arr, i64 0, i64 1
  %p1 = getelementptr [5 x i32], ptr @arr, i64 0, i64 1
  store i32 1, ptr %p0, align 4
  store i32 2, ptr %p1, align 4
  ret void
}
; CHECK-LABEL: define void @mustalias_same_offsets
; CHECK:       call void @__tsan_write4(
; CHECK-NOT:   call void @__tsan_write4(
; CHECK:       ret void

; phi selecting between two globals is MayAlias: no elimination.
define void @mayalias_phi(i1 %c) nounwind uwtable sanitize_thread {
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
; CHECK-LABEL: define void @mayalias_phi
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       call void @__tsan_read4(
; CHECK:       ret void

; ptrtoint/inttoptr round-trip breaks MustAlias: no elimination.
define void @noalias_ptr_roundtrip() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  %i  = ptrtoint ptr @g1 to i64
  %p2 = inttoptr i64 %i to ptr
  %v  = load i32, ptr %p2, align 4
  ret void
}
; CHECK-LABEL: define void @noalias_ptr_roundtrip
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK:       call void @__tsan_read4(
; CHECK:       ret void

; select with identical arms is MustAlias: elimination fires.
define void @mustalias_select_same_ptr(i1 %c) nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  %p = select i1 %c, ptr @g1, ptr @g1
  %v = load i32, ptr %p, align 4
  ret void
}
; CHECK-LABEL: define void @mustalias_select_same_ptr
; CHECK:       call void @__tsan_write4(ptr @g1)
; CHECK-NOT:   call void @__tsan_read4(
; CHECK:       ret void

; ===========================================================================
; Volatile (VOLATILE check prefix only)
;
; When -tsan-distinguish-volatile is active, volatile and non-volatile accesses
; to the same address emit different TSan calls and must never be merged.
; ===========================================================================

; Non-volatile write followed by volatile write: both must be kept.
define void @write_then_volatile_write() nounwind uwtable sanitize_thread {
entry:
  store i32 1, ptr @g1, align 4
  store volatile i32 2, ptr @g1, align 4
  ret void
}
; VOLATILE-LABEL: define void @write_then_volatile_write
; VOLATILE:       call void @__tsan_write4(ptr @g1)
; VOLATILE:       call void @__tsan_volatile_write4(ptr @g1)
; VOLATILE:       ret void

; Volatile write followed by non-volatile write: both must be kept.
define void @volatile_write_then_write() nounwind uwtable sanitize_thread {
entry:
  store volatile i32 1, ptr @g1, align 4
  store i32 2, ptr @g1, align 4
  ret void
}
; VOLATILE-LABEL: define void @volatile_write_then_write
; VOLATILE:       call void @__tsan_volatile_write4(ptr @g1)
; VOLATILE:       call void @__tsan_write4(ptr @g1)
; VOLATILE:       ret void

attributes #0 = { nosync }
