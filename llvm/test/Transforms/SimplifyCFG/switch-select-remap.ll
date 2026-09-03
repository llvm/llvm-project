; RUN: opt < %s -passes=simplifycfg -S | FileCheck %s

; The compared value 4 has no explicit case, and the remapped value 6 maps to
; a real (non-default) case, so switching on %x needs a new explicit case for
; 4 pointing to bb2.
define void @test_remap_add_case(i8 %x) {
; CHECK-LABEL: define void @test_remap_add_case(
; CHECK-SAME: i8 [[X:%.*]]) {
; CHECK-NEXT:    switch i8 [[X]], label [[BB1:%.*]] [
; CHECK-NEXT:      i8 6, label [[BB2:%.*]]
; CHECK-NEXT:      i8 10, label [[BB3:%.*]]
; CHECK-NEXT:      i8 4, label [[BB2]]
; CHECK-NEXT:    ]
;
  %cmp = icmp eq i8 %x, 4
  %key = select i1 %cmp, i8 6, i8 %x
  switch i8 %key, label %bb1 [
    i8 6, label %bb2
    i8 10, label %bb3
  ]

bb1:
  call void @func1()
  unreachable
bb2:
  call void @func2()
  unreachable
bb3:
  call void @func3()
  unreachable
}

; The value 4 already has an explicit case pointing to bb4, but %key can never
; actually be 4 (it's remapped to 6 whenever %x is 4), so that case is really
; dead and should be retargeted to wherever the remapped value 6 dispatches to
; (bb2). Unlike the old InstCombine version of this fold, bb4 and its now-dead
; body are removed in the same run: this pass edits the CFG through a
; DomTreeUpdater, so a follow-up SimplifyCFG iteration cleans it up
; immediately instead of needing a separate pass.
define void @test_remap_retarget_case(i8 %x) {
; CHECK-LABEL: define void @test_remap_retarget_case(
; CHECK-SAME: i8 [[X:%.*]]) {
; CHECK-NEXT:    switch i8 [[X]], label [[BB1:%.*]] [
; CHECK-NEXT:      i8 4, label [[BB2:%.*]]
; CHECK-NEXT:      i8 6, label [[BB2]]
; CHECK-NEXT:      i8 10, label [[BB3:%.*]]
; CHECK-NEXT:    ]
;
  %cmp = icmp eq i8 %x, 4
  %key = select i1 %cmp, i8 6, i8 %x
  switch i8 %key, label %bb1 [
    i8 4, label %bb4
    i8 6, label %bb2
    i8 10, label %bb3
  ]

bb1:
  call void @func1()
  unreachable
bb2:
  call void @func2()
  unreachable
bb3:
  call void @func3()
  unreachable
bb4:
  call void @func4()
  unreachable
}

; Same remap expressed with icmp ne / select(cond, %x, 6). The remapped value 6
; maps to a real (non-default) case, so switching on %x needs a new explicit
; case for 4 pointing to bb2, same as test_remap_add_case but via the NE arm.
define void @test_remap_ne_add_case(i8 %x) {
; CHECK-LABEL: define void @test_remap_ne_add_case(
; CHECK-SAME: i8 [[X:%.*]]) {
; CHECK-NEXT:    switch i8 [[X]], label [[BB1:%.*]] [
; CHECK-NEXT:      i8 6, label [[BB2:%.*]]
; CHECK-NEXT:      i8 10, label [[BB3:%.*]]
; CHECK-NEXT:      i8 4, label [[BB2]]
; CHECK-NEXT:    ]
;
  %cmp = icmp ne i8 %x, 4
  %key = select i1 %cmp, i8 %x, i8 6
  switch i8 %key, label %bb1 [
    i8 6, label %bb2
    i8 10, label %bb3
  ]

bb1:
  call void @func1()
  unreachable
bb2:
  call void @func2()
  unreachable
bb3:
  call void @func3()
  unreachable
}

; Same shape as test_remap_add_case, but the target of the remapped case (bb2)
; has a PHI node. Adding the new case for 4 creates a second edge from this
; block into bb2, so the PHI needs a matching second incoming entry - without
; it this used to hit "Invalid basic block argument to remove!" deep in a
; later pass once something tried to prune one of the two entry edges.
define void @test_remap_add_case_phi(i8 %x, i32 %a) {
; CHECK-LABEL: define void @test_remap_add_case_phi(
; CHECK-SAME: i8 [[X:%.*]], i32 [[A:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i8 [[X]], label [[BB1:%.*]] [
; CHECK-NEXT:      i8 6, label [[BB2:%.*]]
; CHECK-NEXT:      i8 10, label [[BB3:%.*]]
; CHECK-NEXT:      i8 4, label [[BB2]]
; CHECK-NEXT:    ]
; CHECK:       bb2:
; CHECK-NEXT:    [[V:%.*]] = phi i32 [ [[A]], [[ENTRY:%.*]] ], [ [[A]], [[ENTRY]] ]
; CHECK-NEXT:    call void @use(i32 [[V]])
;
entry:
  %cmp = icmp eq i8 %x, 4
  %key = select i1 %cmp, i8 6, i8 %x
  switch i8 %key, label %bb1 [
    i8 6, label %bb2
    i8 10, label %bb3
  ]

bb1:
  call void @func1()
  unreachable
bb2:
  %v = phi i32 [%a, %entry]
  call void @use(i32 %v)
  unreachable
bb3:
  call void @func3()
  unreachable
}

; Same shape as test_remap_retarget_case, but both the remapped case's target
; (bb2) and the retargeted-away-from block (bb4) have PHI nodes: bb2 needs a
; second incoming entry for the new edge, and bb4 (along with its now-stale
; PHI entry) is removed once it becomes unreachable.
define void @test_remap_retarget_case_phi(i8 %x, i32 %a, i32 %c4) {
; CHECK-LABEL: define void @test_remap_retarget_case_phi(
; CHECK-SAME: i8 [[X:%.*]], i32 [[A:%.*]], i32 [[C4:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i8 [[X]], label [[BB1:%.*]] [
; CHECK-NEXT:      i8 4, label [[BB2:%.*]]
; CHECK-NEXT:      i8 6, label [[BB2]]
; CHECK-NEXT:      i8 10, label [[BB3:%.*]]
; CHECK-NEXT:    ]
; CHECK:       bb2:
; CHECK-NEXT:    [[V:%.*]] = phi i32 [ [[A]], [[ENTRY:%.*]] ], [ [[A]], [[ENTRY]] ]
; CHECK-NEXT:    call void @use(i32 [[V]])
;
entry:
  %cmp = icmp eq i8 %x, 4
  %key = select i1 %cmp, i8 6, i8 %x
  switch i8 %key, label %bb1 [
    i8 4, label %bb4
    i8 6, label %bb2
    i8 10, label %bb3
  ]

bb1:
  call void @func1()
  unreachable
bb2:
  %v = phi i32 [%a, %entry]
  call void @use(i32 %v)
  unreachable
bb3:
  call void @func3()
  unreachable
bb4:
  %w = phi i32 [%c4, %entry]
  call void @use(i32 %w)
  unreachable
}

; K (the remapped-to value) has no explicit case of its own, so it already
; dispatches to the default destination - same as the (also absent) compared
; value C would. %x can be switched on directly with no case-list change.
define void @test_remap_k_default_add(i8 %x) {
; CHECK-LABEL: define void @test_remap_k_default_add(
; CHECK-SAME: i8 [[X:%.*]]) {
; CHECK-NEXT:    switch i8 [[X]], label [[DEFAULT:%.*]] [
; CHECK-NEXT:      i8 1, label [[BB1:%.*]]
; CHECK-NEXT:      i8 2, label [[BB2:%.*]]
; CHECK-NEXT:    ]
;
  %cmp = icmp eq i8 %x, 4
  %key = select i1 %cmp, i8 6, i8 %x
  switch i8 %key, label %default [
    i8 1, label %bb1
    i8 2, label %bb2
  ]

bb1:
  call void @func1()
  unreachable
bb2:
  call void @func2()
  unreachable
default:
  call void @func3()
  unreachable
}

; Same as above, but C (4) already has an explicit (and, before the fold,
; unreachable) case pointing to bb4. Since K dispatches to the default
; destination, that stale case must be retargeted there too rather than left
; on bb4 - otherwise a now-reachable X == 4 would wrongly branch to bb4
; instead of falling through to default like case K does. (SimplifyCFG then
; goes on to fold the resulting two-arm switch into a plain icmp/br, which is
; an unrelated, separate simplification.)
define void @test_remap_k_default_retarget(i8 %x) {
; CHECK-LABEL: define void @test_remap_k_default_retarget(
; CHECK-SAME: i8 [[X:%.*]]) {
; CHECK-NEXT:    [[COND:%.*]] = icmp eq i8 [[X]], 1
; CHECK-NEXT:    br i1 [[COND]], label [[BB1:%.*]], label [[DEFAULT:%.*]]
;
  %cmp = icmp eq i8 %x, 4
  %key = select i1 %cmp, i8 6, i8 %x
  switch i8 %key, label %default [
    i8 4, label %bb4
    i8 1, label %bb1
  ]

bb4:
  call void @func4()
  unreachable
bb1:
  call void @func1()
  unreachable
default:
  call void @func3()
  unreachable
}

; The fold changes the case list (a case may be added, or retargeted to a
; different successor), so any existing branch-weight metadata would
; mislabel the new layout - it must be dropped rather than kept stale.
define void @test_remap_drops_branch_weights(i8 %x) {
; CHECK-LABEL: define void @test_remap_drops_branch_weights(
; CHECK-SAME: i8 [[X:%.*]]) {
; CHECK-NEXT:    switch i8 [[X]], label [[BB1:%.*]] [
; CHECK-NEXT:      i8 6, label [[BB2:%.*]]
; CHECK-NEXT:      i8 10, label [[BB3:%.*]]
; CHECK-NEXT:      i8 4, label [[BB2]]
; CHECK-NEXT:    ]
; CHECK-NOT:     !prof
;
  %cmp = icmp eq i8 %x, 4
  %key = select i1 %cmp, i8 6, i8 %x
  switch i8 %key, label %bb1 [
    i8 6, label %bb2
    i8 10, label %bb3
  ], !prof !0

bb1:
  call void @func1()
  unreachable
bb2:
  call void @func2()
  unreachable
bb3:
  call void @func3()
  unreachable
}

; Negative test: %key (the select) is used by more than just the switch, so
; folding it away wouldn't actually remove the compare/select sequence -
; leave it alone.
define void @test_remap_multiuse_select(i8 %x, ptr %p) {
; CHECK-LABEL: define void @test_remap_multiuse_select(
; CHECK-SAME: i8 [[X:%.*]], ptr [[P:%.*]]) {
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[X]], 4
; CHECK-NEXT:    [[KEY:%.*]] = select i1 [[CMP]], i8 6, i8 [[X]]
; CHECK-NEXT:    store i8 [[KEY]], ptr [[P]], align 1
; CHECK-NEXT:    switch i8 [[KEY]], label [[BB1:%.*]] [
; CHECK-NEXT:      i8 6, label [[BB2:%.*]]
; CHECK-NEXT:      i8 10, label [[BB3:%.*]]
; CHECK-NEXT:    ]
;
  %cmp = icmp eq i8 %x, 4
  %key = select i1 %cmp, i8 6, i8 %x
  store i8 %key, ptr %p
  switch i8 %key, label %bb1 [
    i8 6, label %bb2
    i8 10, label %bb3
  ]

bb1:
  call void @func1()
  unreachable
bb2:
  call void @func2()
  unreachable
bb3:
  call void @func3()
  unreachable
}

; The compare (%cmp) is used by more than just the select, but that's fine -
; only the select itself needs to be single-use for the fold to remove it;
; %cmp is left alone for its other use.
define void @test_remap_multiuse_icmp(i8 %x, ptr %p) {
; CHECK-LABEL: define void @test_remap_multiuse_icmp(
; CHECK-SAME: i8 [[X:%.*]], ptr [[P:%.*]]) {
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[X]], 4
; CHECK-NEXT:    store i1 [[CMP]], ptr [[P]], align 1
; CHECK-NEXT:    switch i8 [[X]], label [[BB1:%.*]] [
; CHECK-NEXT:      i8 6, label [[BB2:%.*]]
; CHECK-NEXT:      i8 10, label [[BB3:%.*]]
; CHECK-NEXT:      i8 4, label [[BB2]]
; CHECK-NEXT:    ]
;
  %cmp = icmp eq i8 %x, 4
  store i1 %cmp, ptr %p
  %key = select i1 %cmp, i8 6, i8 %x
  switch i8 %key, label %bb1 [
    i8 6, label %bb2
    i8 10, label %bb3
  ]

bb1:
  call void @func1()
  unreachable
bb2:
  call void @func2()
  unreachable
bb3:
  call void @func3()
  unreachable
}

; Negative test: the select's non-constant arm (%y) doesn't match the icmp's
; non-constant operand (%x), so this isn't a same-value remap and must not
; be folded.
define void @test_remap_mismatched_operand(i8 %x, i8 %y) {
; CHECK-LABEL: define void @test_remap_mismatched_operand(
; CHECK-SAME: i8 [[X:%.*]], i8 [[Y:%.*]]) {
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[X]], 4
; CHECK-NEXT:    [[KEY:%.*]] = select i1 [[CMP]], i8 6, i8 [[Y]]
; CHECK-NEXT:    switch i8 [[KEY]], label [[BB1:%.*]] [
; CHECK-NEXT:      i8 6, label [[BB2:%.*]]
; CHECK-NEXT:      i8 10, label [[BB3:%.*]]
; CHECK-NEXT:    ]
;
  %cmp = icmp eq i8 %x, 4
  %key = select i1 %cmp, i8 6, i8 %y
  switch i8 %key, label %bb1 [
    i8 6, label %bb2
    i8 10, label %bb3
  ]

bb1:
  call void @func1()
  unreachable
bb2:
  call void @func2()
  unreachable
bb3:
  call void @func3()
  unreachable
}

declare void @func1()
declare void @func2()
declare void @func3()
declare void @func4()
declare void @use(i32)

!0 = !{!"branch_weights", i32 1, i32 2, i32 3}
