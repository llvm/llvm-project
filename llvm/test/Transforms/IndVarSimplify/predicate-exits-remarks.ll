; Check that IndVarSimplify::predicateLoopExits emits optimization remarks
; with structured ore::NV fields for tool consumption.
;
; Textual remark stream:
; RUN: opt -passes=indvars -pass-remarks=indvars -pass-remarks-missed=indvars \
; RUN:     -disable-output %s 2>&1 | FileCheck %s
;
; YAML remark stream (structured fields):
; RUN: opt -passes=indvars -pass-remarks-output=%t.yaml -disable-output %s
; RUN: FileCheck --check-prefix=YAML %s < %t.yaml

; CHECK: remark: {{.*}}: Exit count computable for exit
; CHECK: remark: {{.*}}: Loop exit predicated and hoisted to preheader
; CHECK: remark: {{.*}}: Exit count not computable for exit {{.*}} trapExit=true{{.*}} normalExit=false{{.*}} reason=unknown
; CHECK: remark: {{.*}}: Exit count not computable for exit {{.*}} trapExit=false{{.*}} normalExit=true{{.*}} reason=unknown
; CHECK: remark: {{.*}}: Unable to predicate loop exits: could not compute an exact trip count
; CHECK: remark: {{.*}}: Exit count not computable for exit {{.*}} reason=not_dominating_latch
; CHECK: remark: {{.*}}: Unable to predicate loop exits: could not compute an exact trip count
; CHECK: remark: {{.*}}: Unable to predicate loop exit: exit is not a conditional branch
; CHECK: remark: {{.*}}: Unable to predicate loop exits: no predicatable exit remaining
; CHECK: remark: {{.*}}: Unable to predicate loop exit: exiting block belongs to a different (inner) loop
; CHECK: remark: {{.*}}: Unable to predicate loop exit: values computed inside the loop are used after the exit
; CHECK: remark: {{.*}}: Unable to predicate loop exit: trap block may observe side effects
; CHECK: remark: {{.*}}: Unable to predicate loop exits: loop contains an atomic or volatile store
; CHECK: remark: {{.*}}: Unable to predicate loop exits: loop body has side effects
; CHECK: remark: {{.*}}: Exit count computable for exit
; CHECK: remark: {{.*}}: Loop exit predicated and hoisted to preheader

; CHECK-NOT: loop_with_annotated_call{{.*}}Unable to predicate loop exits: loop body has side effects

; YAML-DAG: Name:            ExitCountComputable
; YAML-DAG: Name:            PredicatedExit
; YAML-DAG: Name:            ExitCountNotComputable
; YAML-DAG: Name:            ExactBTCUnknown
; YAML-DAG: Name:            NonBranchExit
; YAML-DAG: Name:            NoPredicatableExits
; YAML-DAG: Name:            ExitNotInCurrentLoop
; YAML-DAG: Name:            ExitBlockHasPhis
; YAML-DAG: Name:            TrapBlockObservesSideEffects
; YAML-DAG: Name:            AtomicStoreInLoop
; YAML-DAG: Name:            LoopSideEffects
; YAML-DAG:   - IsTrapExit:      'true'
; YAML-DAG:   - Predicate:       ugt
; YAML-DAG:   - Stride:          '+4'
; YAML-DAG:   - Reason:          not_dominating_latch
; YAML-DAG:   - Reason:          unknown

;----------------------------------------------------------------------------
; Trivially predicatable single-exit loop -> ExitCountComputable + PredicatedExit.
;----------------------------------------------------------------------------
define void @predicatable(i32 %n) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nuw nsw i32 %iv, 1
  %cmp = icmp ult i32 %iv.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

;----------------------------------------------------------------------------
; Multi-exit with a trap exit. The trap-exit's count is not computable (no
; wrap flags on the address AddRec), so we expect ExactBTCUnknown on the
; header and an ExitCountNotComputable with IsTrapExit=true.
;----------------------------------------------------------------------------
define void @multi_exit_with_trap(i64 %start, i64 %end, i64 %limit) mustprogress {
entry:
  %cmp.entry = icmp ule i64 %start, %end
  br i1 %cmp.entry, label %loop, label %exit
loop:
  %iv = phi i64 [ %start, %entry ], [ %iv.next, %latch ]
  %trap.cmp = icmp ugt i64 %iv, %limit
  br i1 %trap.cmp, label %trap, label %latch
latch:
  %iv.next = add i64 %iv, 4
  %cond = icmp ne i64 %iv.next, %end
  br i1 %cond, label %loop, label %exit
trap:
  call void @llvm.trap()
  unreachable
exit:
  ret void
}

;----------------------------------------------------------------------------
; Two exiting blocks; `inner` does not dominate the latch because its
; successor `merge` is also reachable via `other_side`. SCEV cannot compute
; an exit count for a non-dominating exit (see
; ScalarEvolution::computeExitLimit's dominance guard), so this gets a
; not-computable remark with Reason=not_dominating_latch. The overall bail
; is ExactBTCUnknown (one exit missing makes the BTC unknown).
;----------------------------------------------------------------------------
define void @exit_not_dominating_latch(i32 %n) {
entry:
  br label %header
header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %cond1 = icmp ult i32 %iv, 100
  br i1 %cond1, label %split, label %exit1
split:
  %pick = icmp ult i32 %iv, %n
  br i1 %pick, label %inner, label %other_side
inner:
  %cond_exit = icmp eq i32 %iv, 5
  br i1 %cond_exit, label %exit2, label %merge
other_side:
  br label %merge
merge:
  br label %latch
latch:
  %iv.next = add nuw nsw i32 %iv, 1
  br label %header
exit1:
  ret void
exit2:
  ret void
}

;----------------------------------------------------------------------------
; Sole exit is a `switch` -> BadExit filters it (NonBranchExit), leaving
; ExitingBlocks empty -> NoPredicatableExits fires. SCEV handles the
; single-exit switch via computeExitLimitFromSingleExitSwitch, so ExactBTC
; is known (10) and we actually reach the filter phase.
;----------------------------------------------------------------------------
define void @all_exits_filtered_via_switch() {
entry:
  br label %header
header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %header ]
  %iv.next = add nuw nsw i32 %iv, 1
  switch i32 %iv, label %header [ i32 10, label %exit ]
exit:
  ret void
}

;----------------------------------------------------------------------------
; Nested loops; when processing the OUTER loop, inner.header is an exiting
; block of outer (branches to exit_outer) but LI->getLoopFor(inner.header)
; returns the INNER loop != outer -> BadExit fires ExitNotInCurrentLoop.
; Both outer exits have SCEV-computable counts (so ExactBTC is known).
;----------------------------------------------------------------------------
define void @nested_exit_from_inner() mustprogress {
entry:
  br label %outer.header
outer.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner.header
inner.header:
  %j = phi i32 [ 0, %outer.header ], [ %j.next, %inner.latch ]
  %early = icmp eq i32 %j, 5
  br i1 %early, label %exit_outer, label %inner.latch
inner.latch:
  %j.next = add nuw nsw i32 %j, 1
  %inner_cond = icmp ult i32 %j.next, 10
  br i1 %inner_cond, label %inner.header, label %outer.latch
outer.latch:
  %i.next = add nuw nsw i32 %i, 1
  %outer_cond = icmp ult i32 %i.next, 100
  br i1 %outer_cond, label %outer.header, label %exit_outer
exit_outer:
  ret void
}

;----------------------------------------------------------------------------
; One exit block holds an LCSSA phi consuming an in-loop value that's used
; by the caller. BadExit filters that exiting block -> ExitBlockHasPhis.
; The other exit is clean so the filter path fires cleanly.
;----------------------------------------------------------------------------
define i32 @exit_block_has_lcssa_phi(i32 %n, ptr %p) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.body ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %loop.body ]
  %cmp.early = icmp ult i32 %iv, %n
  br i1 %cmp.early, label %loop.body, label %exit.phi
loop.body:
  %ld = load i32, ptr %p, align 4
  %acc.next = add i32 %acc, %ld
  %iv.next = add nuw nsw i32 %iv, 1
  %cmp.latch = icmp ult i32 %iv.next, 100
  br i1 %cmp.latch, label %loop, label %exit.clean
exit.phi:
  %lcssa = phi i32 [ %acc, %loop ]
  %use = mul i32 %lcssa, 7
  ret i32 %use
exit.clean:
  ret i32 0
}

;----------------------------------------------------------------------------
; Loop has a simple store (HasThreadLocalSideEffects=true). Its exit targets
; a block ending in unreachable that also contains an extra store, so
; crashingBBWithoutEffect rejects it -> TrapBlockObservesSideEffects fires.
;----------------------------------------------------------------------------
define void @trap_block_observes_side_effects(ptr %p, ptr %q) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  store i32 %iv, ptr %p, align 4
  %iv.next = add nuw nsw i32 %iv, 1
  %cond = icmp ult i32 %iv.next, 100
  br i1 %cond, label %loop, label %trap
trap:
  store i32 0, ptr %q, align 4
  unreachable
}

;----------------------------------------------------------------------------
; Loop containing an atomic store -> AtomicStoreInLoop missed remark.
;----------------------------------------------------------------------------
define void @loop_with_atomic_store(ptr %p, i32 %n) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  store atomic i32 %iv, ptr %p seq_cst, align 4
  %iv.next = add nuw nsw i32 %iv, 1
  %cmp = icmp ult i32 %iv.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

;----------------------------------------------------------------------------
; Loop containing a call with side effects -> LoopSideEffects missed remark.
;----------------------------------------------------------------------------
declare void @side_effect()

define void @loop_with_side_effect_call(i32 %n) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  call void @side_effect()
  %iv.next = add nuw nsw i32 %iv, 1
  %cmp = icmp ult i32 %iv.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

declare void @llvm.trap()

;----------------------------------------------------------------------------
; Contrast: a call whose declaration carries memory(none)+nounwind+willreturn
; has mayHaveSideEffects()==false, so predicateLoopExits does NOT bail at the
; side-effects check. The loop should be predicated just like @predicatable.
;
; In C source, these attributes come from:
;   __attribute__((const))    -> memory(none) nounwind willreturn
;   __attribute__((pure))     -> memory(read) nounwind willreturn   (also OK)
; Either GCC/Clang attribute makes the callee appear side-effect-free to LLVM
; (see Instruction::mayHaveSideEffects = mayWriteToMemory || mayThrow ||
;  !willReturn), which in turn lets IndVarSimplify predicate the exit.
;----------------------------------------------------------------------------
declare void @no_side_effect_call() memory(none) nounwind willreturn

define void @loop_with_annotated_call(i32 %n) {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  call void @no_side_effect_call()
  %iv.next = add nuw nsw i32 %iv, 1
  %cmp = icmp ult i32 %iv.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}
