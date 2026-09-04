; Behavior tests for independent inner-subnest depth and candidate-count
; policies. This file is hand-maintained; do NOT run update_test_checks.py.
;
; A maximum depth of 2 leaves the depth-3 nest unchanged. Separately, a maximum
; of two candidate attempts in a supported depth-3 forest leaves all three
; unknown-context pairs unchanged and emits a stable budget-exhaustion remark.
; A companion positive fixture shows that a whole nest that is genuinely too
; deep, but that contains a shallower eligible pair, is interchanged with no
; contradictory "unsupported depth" remark. A separate one-loop fixture checks
; the below-minimum diagnostic. A final fixture with more eligible candidates
; than the budget allows, at four distinct depths, exercises the
; candidate-retention eviction path.
;
; RUN: llvm-extract -S -func=over_cap_depth_filters_inner_pair %s \
; RUN:     -o %t.over-cap
; RUN: opt < %t.over-cap -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-max-loop-nest-depth=2 \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -pass-remarks-output=%t.over-cap.yaml -disable-output
; RUN: FileCheck %s --check-prefix=OVER-CAP \
; RUN:     --input-file=%t.over-cap.yaml --implicit-check-not=Interchanged
; RUN: opt < %t.over-cap -passes='loop(loop-interchange),print<loops>' \
; RUN:     -cache-line-size=64 -loop-interchange-max-loop-nest-depth=2 \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=OVER-CAP-LOOPS
;
; RUN: llvm-extract -S -func=budget_exhausted_all_fail %s -o %t.budget
; RUN: opt < %t.budget -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-max-loop-nest-depth=3 \
; RUN:     -loop-interchange-max-inner-subnest-candidates=2 \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -pass-remarks-output=%t.budget.yaml -disable-output
; RUN: FileCheck %s --check-prefix=BUDGET --input-file=%t.budget.yaml \
; RUN:     --implicit-check-not=Interchanged
; RUN: opt < %t.budget -passes='loop(loop-interchange),print<loops>' \
; RUN:     -cache-line-size=64 -loop-interchange-max-loop-nest-depth=3 \
; RUN:     -loop-interchange-max-inner-subnest-candidates=2 \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=BUDGET-LOOPS
;
; RUN: llvm-extract -S -func=too_deep_reaches_shallow_pair %s -o %t.too-deep
; RUN: opt < %t.too-deep -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-max-loop-nest-depth=3 \
; RUN:     -loop-interchange-profitabilities=ignore \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -pass-remarks-output=%t.too-deep.yaml -disable-output
; RUN: FileCheck %s --check-prefix=TOO-DEEP --input-file=%t.too-deep.yaml \
; RUN:     --implicit-check-not=Dependence \
; RUN:     --implicit-check-not=UnsupportedLoopNestDepth
; RUN: opt < %t.too-deep -passes='loop(loop-interchange),print<loops>' \
; RUN:     -cache-line-size=64 -loop-interchange-max-loop-nest-depth=3 \
; RUN:     -loop-interchange-profitabilities=ignore \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=TOO-DEEP-LOOPS
; RUN: opt < %t.too-deep -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-max-loop-nest-depth=3 \
; RUN:     -loop-interchange-enable-inner-subnest-fallback=false \
; RUN:     -loop-interchange-profitabilities=ignore \
; RUN:     -pass-remarks-output=%t.fallback-disabled.yaml -disable-output
; RUN: FileCheck %s --check-prefix=FALLBACK-DISABLED \
; RUN:     --input-file=%t.fallback-disabled.yaml \
; RUN:     --implicit-check-not=Interchanged --implicit-check-not=Fallback
; RUN: opt < %t.too-deep -passes='loop(loop-interchange),print<loops>' \
; RUN:     -cache-line-size=64 -loop-interchange-max-loop-nest-depth=3 \
; RUN:     -loop-interchange-enable-inner-subnest-fallback=false \
; RUN:     -loop-interchange-profitabilities=ignore \
; RUN:     -disable-output 2>&1 | \
; RUN:     FileCheck %s --check-prefix=FALLBACK-DISABLED-LOOPS
;
; RUN: llvm-extract -S -func=too_shallow_emits_depth_remark %s \
; RUN:     -o %t.too-shallow
; RUN: opt < %t.too-shallow -passes=loop-interchange \
; RUN:     -pass-remarks-output=%t.too-shallow.yaml -disable-output
; RUN: FileCheck %s --check-prefix=TOO-SHALLOW \
; RUN:     --input-file=%t.too-shallow.yaml
;
; RUN: llvm-extract -S -func=eviction_prunes_shallowest_surplus %s -o %t.evict
; RUN: opt < %t.evict -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-max-inner-subnest-candidates=2 \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -pass-remarks-output=%t.evict.yaml -disable-output
; RUN: FileCheck %s --check-prefix=EVICT --input-file=%t.evict.yaml \
; RUN:     --implicit-check-not=Interchanged
; RUN: opt < %t.evict -passes='loop(loop-interchange),print<loops>' \
; RUN:     -cache-line-size=64 -loop-interchange-max-inner-subnest-candidates=2 \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=EVICT-LOOPS
; RUN: opt < %t.evict -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-max-loop-nest-depth=5 \
; RUN:     -loop-interchange-max-inner-subnest-candidates=2 \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -pass-remarks-output=%t.too-deep-fail.yaml -disable-output
; RUN: FileCheck %s --check-prefix=TOO-DEEP-FAIL \
; RUN:     --input-file=%t.too-deep-fail.yaml \
; RUN:     --implicit-check-not=Interchanged

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@CF1 = global [8 x [8 x double]] zeroinitializer
@CF2 = global [8 x [8 x double]] zeroinitializer
@CF3 = global [8 x [8 x double]] zeroinitializer
@A = global [8 x [8 x [8 x double]]] zeroinitializer
@B = global [8 x [8 x [8 x double]]] zeroinitializer
@DA = global [8 x [8 x double]] zeroinitializer
@DB = global [8 x [8 x double]] zeroinitializer
@DC = global [8 x [8 x double]] zeroinitializer
@DD = global [8 x [8 x double]] zeroinitializer

; OVER-CAP:      --- !Missed
; OVER-CAP:      Name:            UnsupportedLoopNestDepth
; OVER-CAP-NEXT: Function:        over_cap_depth_filters_inner_pair
;
; BUDGET-NOT:  --- !
; BUDGET:      --- !Missed
; BUDGET-NEXT: Pass:            loop-interchange
; BUDGET-NEXT: Name:            FallbackUnknownContext
; BUDGET-NEXT: DebugLoc:        { File: inner-subnest-budget.ll, Line: 70, Column: 1 }
; BUDGET-NEXT: Function:        budget_exhausted_all_fail
; BUDGET-NEXT: Args:
; BUDGET-NEXT:   - String:          'Cannot interchange inner subnest: the surrounding dependence context is unknown or unsafe.'
; BUDGET-NEXT: ...
; BUDGET-NEXT: --- !Missed
; BUDGET-NEXT: Pass:            loop-interchange
; BUDGET-NEXT: Name:            FallbackUnknownContext
; BUDGET-NEXT: DebugLoc:        { File: inner-subnest-budget.ll, Line: 80, Column: 1 }
; BUDGET-NEXT: Function:        budget_exhausted_all_fail
; BUDGET-NEXT: Args:
; BUDGET-NEXT:   - String:          'Cannot interchange inner subnest: the surrounding dependence context is unknown or unsafe.'
; BUDGET-NEXT: ...
; BUDGET-NEXT: --- !Missed
; BUDGET-NEXT: Pass:            loop-interchange
; BUDGET-NEXT: Name:            FallbackCandidateBudget
; BUDGET-NEXT: DebugLoc:        { File: inner-subnest-budget.ll, Line: 90, Column: 1 }
; BUDGET-NEXT: Function:        budget_exhausted_all_fail
; BUDGET-NEXT: Args:
; BUDGET-NEXT:   - String:          'Inner-subnest candidate budget exhausted; the loop nest is left unchanged.'
; BUDGET-NEXT: ...
; BUDGET-NOT:  --- !
;
; The whole nest is too deep, but no "unsupported depth" remark appears because
; the shallower pair is found and interchanged; only a Passed record fires.
; The implicit exclusions also reject a supported-depth analysis record.
; TOO-DEEP-NOT:  --- !
; TOO-DEEP:      --- !Passed
; TOO-DEEP-NEXT: Pass:            loop-interchange
; TOO-DEEP-NEXT: Name:            Interchanged
; TOO-DEEP-NEXT: Function:        too_deep_reaches_shallow_pair
; TOO-DEEP-NEXT: Args:
; TOO-DEEP-NEXT:   - String:          Loop interchanged with enclosing loop.
; TOO-DEEP-NEXT: ...
; TOO-DEEP-NOT:  --- !
;
; FALLBACK-DISABLED-NOT:  --- !
; FALLBACK-DISABLED:      --- !Missed
; FALLBACK-DISABLED-NEXT: Pass:            loop-interchange
; FALLBACK-DISABLED-NEXT: Name:            UnsupportedLoopNestDepth
; FALLBACK-DISABLED-NEXT: Function:        too_deep_reaches_shallow_pair
; FALLBACK-DISABLED-NEXT: Args:
; FALLBACK-DISABLED-NEXT:   - String:          'Unsupported depth of loop nest, the supported range is ['
; FALLBACK-DISABLED-NEXT:   - String:          '2'
; FALLBACK-DISABLED-NEXT:   - String:          ', '
; FALLBACK-DISABLED-NEXT:   - String:          '3'
; FALLBACK-DISABLED-NEXT:   - String:          "].\n"
; FALLBACK-DISABLED-NEXT: ...
; FALLBACK-DISABLED-NOT:  --- !
;
; TOO-SHALLOW-NOT:  --- !
; TOO-SHALLOW:      --- !Missed
; TOO-SHALLOW-NEXT: Pass:            loop-interchange
; TOO-SHALLOW-NEXT: Name:            UnsupportedLoopNestDepth
; TOO-SHALLOW-NEXT: Function:        too_shallow_emits_depth_remark
; TOO-SHALLOW-NEXT: Args:
; TOO-SHALLOW-NEXT:   - String:          'Unsupported depth of loop nest, the supported range is ['
; TOO-SHALLOW-NEXT:   - String:          '2'
; TOO-SHALLOW-NEXT:   - String:          ', '
; TOO-SHALLOW-NEXT:   - String:          '10'
; TOO-SHALLOW-NEXT:   - String:          "].\n"
; TOO-SHALLOW-NEXT: ...
; TOO-SHALLOW-NOT:  --- !
;
; Deepest-first: the two deepest retained candidates (depth 6, then depth 5)
; are attempted and rejected; the depth-4 candidate is retained only to anchor
; the overflow remark and is never attempted. The depth-3 candidate is evicted.
; Distinct debug locations identify each candidate and exact NEXT checks reject
; additional attempts.
; EVICT-NOT:  --- !
; EVICT:      --- !Missed
; EVICT-NEXT: Pass:            loop-interchange
; EVICT-NEXT: Name:            FallbackUnknownContext
; EVICT-NEXT: DebugLoc:        { File: inner-subnest-budget.ll, Line: 60, Column: 1 }
; EVICT-NEXT: Function:        eviction_prunes_shallowest_surplus
; EVICT-NEXT: Args:
; EVICT-NEXT:   - String:          'Cannot interchange inner subnest: the surrounding dependence context is unknown or unsafe.'
; EVICT-NEXT: ...
; EVICT-NEXT: --- !Missed
; EVICT-NEXT: Pass:            loop-interchange
; EVICT-NEXT: Name:            FallbackUnknownContext
; EVICT-NEXT: DebugLoc:        { File: inner-subnest-budget.ll, Line: 50, Column: 1 }
; EVICT-NEXT: Function:        eviction_prunes_shallowest_surplus
; EVICT-NEXT: Args:
; EVICT-NEXT:   - String:          'Cannot interchange inner subnest: the surrounding dependence context is unknown or unsafe.'
; EVICT-NEXT: ...
; EVICT-NEXT: --- !Missed
; EVICT-NEXT: Pass:            loop-interchange
; EVICT-NEXT: Name:            FallbackCandidateBudget
; EVICT-NEXT: DebugLoc:        { File: inner-subnest-budget.ll, Line: 40, Column: 1 }
; EVICT-NEXT: Function:        eviction_prunes_shallowest_surplus
; EVICT-NEXT: Args:
; EVICT-NEXT:   - String:          'Inner-subnest candidate budget exhausted; the loop nest is left unchanged.'
; EVICT-NEXT: ...
; EVICT-NOT:  --- !
;
; With maximum depth 5, the depth-6 candidate is filtered during enumeration.
; The depth-5 and depth-4 candidates are attempted and rejected, the depth-3
; candidate anchors the budget remark, and the final record reports the
; unsupported whole-nest depth.
; TOO-DEEP-FAIL-NOT:  --- !
; TOO-DEEP-FAIL:      --- !Missed
; TOO-DEEP-FAIL-NEXT: Pass:            loop-interchange
; TOO-DEEP-FAIL-NEXT: Name:            FallbackUnknownContext
; TOO-DEEP-FAIL-NEXT: DebugLoc:        { File: inner-subnest-budget.ll, Line: 50, Column: 1 }
; TOO-DEEP-FAIL-NEXT: Function:        eviction_prunes_shallowest_surplus
; TOO-DEEP-FAIL-NEXT: Args:
; TOO-DEEP-FAIL-NEXT:   - String:          'Cannot interchange inner subnest: the surrounding dependence context is unknown or unsafe.'
; TOO-DEEP-FAIL-NEXT: ...
; TOO-DEEP-FAIL-NEXT: --- !Missed
; TOO-DEEP-FAIL-NEXT: Pass:            loop-interchange
; TOO-DEEP-FAIL-NEXT: Name:            FallbackUnknownContext
; TOO-DEEP-FAIL-NEXT: DebugLoc:        { File: inner-subnest-budget.ll, Line: 40, Column: 1 }
; TOO-DEEP-FAIL-NEXT: Function:        eviction_prunes_shallowest_surplus
; TOO-DEEP-FAIL-NEXT: Args:
; TOO-DEEP-FAIL-NEXT:   - String:          'Cannot interchange inner subnest: the surrounding dependence context is unknown or unsafe.'
; TOO-DEEP-FAIL-NEXT: ...
; TOO-DEEP-FAIL-NEXT: --- !Missed
; TOO-DEEP-FAIL-NEXT: Pass:            loop-interchange
; TOO-DEEP-FAIL-NEXT: Name:            FallbackCandidateBudget
; TOO-DEEP-FAIL-NEXT: DebugLoc:        { File: inner-subnest-budget.ll, Line: 30, Column: 1 }
; TOO-DEEP-FAIL-NEXT: Function:        eviction_prunes_shallowest_surplus
; TOO-DEEP-FAIL-NEXT: Args:
; TOO-DEEP-FAIL-NEXT:   - String:          'Inner-subnest candidate budget exhausted; the loop nest is left unchanged.'
; TOO-DEEP-FAIL-NEXT: ...
; TOO-DEEP-FAIL-NEXT: --- !Missed
; TOO-DEEP-FAIL-NEXT: Pass:            loop-interchange
; TOO-DEEP-FAIL-NEXT: Name:            UnsupportedLoopNestDepth
; TOO-DEEP-FAIL-NEXT: Function:        eviction_prunes_shallowest_surplus
; TOO-DEEP-FAIL-NEXT: Args:
; TOO-DEEP-FAIL-NEXT:   - String:          'Unsupported depth of loop nest, the supported range is ['
; TOO-DEEP-FAIL-NEXT:   - String:          '2'
; TOO-DEEP-FAIL-NEXT:   - String:          ', '
; TOO-DEEP-FAIL-NEXT:   - String:          '5'
; TOO-DEEP-FAIL-NEXT:   - String:          "].\n"
; TOO-DEEP-FAIL-NEXT: ...
; TOO-DEEP-FAIL-NOT:  --- !

;-------------------------------------------------------------------------------
; The one-loop nest is below the supported minimum and has no fallback path.
; The pass must retain its user-visible unsupported-depth remark.
;-------------------------------------------------------------------------------
define void @too_shallow_emits_depth_remark(ptr %A) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds double, ptr %A, i64 %i
  store double 1.000000e+00, ptr %p, align 8
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i.next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

;-------------------------------------------------------------------------------
; A linear, computable depth-3 nest. The cap (2) rejects the whole nest and its
; depth-3 candidate pair, so no fallback interchange is attempted.
;-------------------------------------------------------------------------------
define void @over_cap_depth_filters_inner_pair(ptr %A, ptr %R) {
entry:
  br label %anc.header

anc.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %anc.latch ]
  br label %pair.outer.header

pair.outer.header:
  %i = phi i64 [ 0, %anc.header ], [ %i.next, %pair.outer.latch ]
  %sum.i = phi double [ 0.000000e+00, %anc.header ], [ %sum.i.lcssa, %pair.outer.latch ]
  br label %pair.inner

pair.inner:
  %j = phi i64 [ 0, %pair.outer.header ], [ %j.next, %pair.inner ]
  %sum.j = phi double [ %sum.i, %pair.outer.header ], [ %sum.j.next, %pair.inner ]
  %idx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  %a = load double, ptr %idx, align 8
  %sum.j.next = fadd reassoc double %sum.j, %a
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %pair.outer.latch, label %pair.inner

pair.outer.latch:
  %sum.i.lcssa = phi double [ %sum.j.next, %pair.inner ]
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %anc.latch, label %pair.outer.header

anc.latch:
  %sum.live = phi double [ %sum.i.lcssa, %pair.outer.latch ]
  store double %sum.live, ptr %R, align 8
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 4
  br i1 %k.ec, label %exit, label %anc.header

exit:
  ret void
}

; The over-cap pair remains in its original order.
; OVER-CAP-LOOPS-LABEL: Loop info for function 'over_cap_depth_filters_inner_pair':
; OVER-CAP-LOOPS:         Loop at depth 1 containing: %anc.header<header>
; OVER-CAP-LOOPS-NEXT:      Loop at depth 2 containing: %pair.outer.header<header>
; OVER-CAP-LOOPS-NEXT:        Loop at depth 3 containing: %pair.inner<header>

; Budget exhaustion leaves all three candidate pairs unswapped and in their
; original sibling order: each pN.i header remains outer to its pN.j body.
; BUDGET-LOOPS-LABEL: Loop info for function 'budget_exhausted_all_fail':
; BUDGET-LOOPS:         Loop at depth 1 containing: %k.header<header>
; BUDGET-LOOPS-NEXT:      Loop at depth 2 containing: %p1.i.header<header>
; BUDGET-LOOPS-NEXT:        Loop at depth 3 containing: %p1.j.body<header>
; BUDGET-LOOPS-NEXT:      Loop at depth 2 containing: %p2.i.header<header>
; BUDGET-LOOPS-NEXT:        Loop at depth 3 containing: %p2.j.body<header>
; BUDGET-LOOPS-NEXT:      Loop at depth 2 containing: %p3.i.header<header>
; BUDGET-LOOPS-NEXT:        Loop at depth 3 containing: %p3.j.body<header>

;-------------------------------------------------------------------------------
; Three eligible sibling pairs under a common ancestor k that does not index the
; arrays, so each pair's surrounding (k) dependence direction is unknown (`*`).
; Every pair is therefore rejected (FallbackUnknownContext). With the budget at
; 2, only two are attempted deepest-first; because a third eligible pair remained
; and all attempts failed, the budget-exhaustion remark fires.
;-------------------------------------------------------------------------------
define void @budget_exhausted_all_fail() !dbg !10 {
entry:
  br label %k.header

k.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %k.latch ]
  br label %p1.i.header

p1.i.header:
  %i1 = phi i64 [ 0, %k.header ], [ %i1.next, %p1.i.latch ]
  br label %p1.j.body, !dbg !11

p1.j.body:
  %j1 = phi i64 [ 1, %p1.i.header ], [ %j1.next, %p1.j.body ]
  %j1m1 = sub i64 %j1, 1
  %p1.ld = getelementptr inbounds [8 x [8 x double]], ptr @CF1, i64 0, i64 %j1, i64 %i1
  %p1.v = load double, ptr %p1.ld, align 8
  %p1.nv = fadd double %p1.v, 1.000000e+00
  %p1.st = getelementptr inbounds [8 x [8 x double]], ptr @CF1, i64 0, i64 %j1m1, i64 %i1
  store double %p1.nv, ptr %p1.st, align 8
  %j1.next = add i64 %j1, 1
  %j1.ec = icmp eq i64 %j1.next, 8
  br i1 %j1.ec, label %p1.i.latch, label %p1.j.body

p1.i.latch:
  %i1.next = add i64 %i1, 1
  %i1.ec = icmp eq i64 %i1.next, 7
  br i1 %i1.ec, label %p2.i.header, label %p1.i.header

p2.i.header:
  %i2 = phi i64 [ 0, %p1.i.latch ], [ %i2.next, %p2.i.latch ]
  br label %p2.j.body, !dbg !12

p2.j.body:
  %j2 = phi i64 [ 1, %p2.i.header ], [ %j2.next, %p2.j.body ]
  %j2m1 = sub i64 %j2, 1
  %p2.ld = getelementptr inbounds [8 x [8 x double]], ptr @CF2, i64 0, i64 %j2, i64 %i2
  %p2.v = load double, ptr %p2.ld, align 8
  %p2.nv = fadd double %p2.v, 1.000000e+00
  %p2.st = getelementptr inbounds [8 x [8 x double]], ptr @CF2, i64 0, i64 %j2m1, i64 %i2
  store double %p2.nv, ptr %p2.st, align 8
  %j2.next = add i64 %j2, 1
  %j2.ec = icmp eq i64 %j2.next, 8
  br i1 %j2.ec, label %p2.i.latch, label %p2.j.body

p2.i.latch:
  %i2.next = add i64 %i2, 1
  %i2.ec = icmp eq i64 %i2.next, 7
  br i1 %i2.ec, label %p3.i.header, label %p2.i.header

p3.i.header:
  %i3 = phi i64 [ 0, %p2.i.latch ], [ %i3.next, %p3.i.latch ]
  br label %p3.j.body, !dbg !13

p3.j.body:
  %j3 = phi i64 [ 1, %p3.i.header ], [ %j3.next, %p3.j.body ]
  %j3m1 = sub i64 %j3, 1
  %p3.ld = getelementptr inbounds [8 x [8 x double]], ptr @CF3, i64 0, i64 %j3, i64 %i3
  %p3.v = load double, ptr %p3.ld, align 8
  %p3.nv = fadd double %p3.v, 1.000000e+00
  %p3.st = getelementptr inbounds [8 x [8 x double]], ptr @CF3, i64 0, i64 %j3m1, i64 %i3
  store double %p3.nv, ptr %p3.st, align 8
  %j3.next = add i64 %j3, 1
  %j3.ec = icmp eq i64 %j3.next, 8
  br i1 %j3.ec, label %p3.i.latch, label %p3.j.body

p3.i.latch:
  %i3.next = add i64 %i3, 1
  %i3.ec = icmp eq i64 %i3.next, 7
  br i1 %i3.ec, label %k.latch, label %p3.i.header

k.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 7
  br i1 %k.ec, label %exit, label %k.header

exit:
  ret void
}

;-------------------------------------------------------------------------------
; A whole nest that is genuinely too deep (true depth 4 > cap 3) has two
; independently-nested subtrees under a common depth-1 ancestor hk: hi/bx at
; true depth 3, and ha/hb/bc at true depth 4. Only the depth-3 hi/bx pair is
; within the cap and is interchanged through the fallback; the deeper ha/hb/bc
; triple has no eligible parent/leaf-child pair within the cap (its only
; candidate, hb/bc, is itself too deep at depth 4) and is left unchanged. The
; whole-nest depth check fails, but because the fallback rescues a shallower
; pair, no UnsupportedLoopNestDepth remark may accompany the Interchanged one.
;-------------------------------------------------------------------------------
define void @too_deep_reaches_shallow_pair() {
entry:
  br label %hk
hk:
  %k = phi i64 [0, %entry], [%kn, %lk]
  br label %hi
hi:
  %i = phi i64 [0, %hk], [%in, %li]
  br label %bx
bx:
  %x = phi i64 [0, %hi], [%xn, %bx]
  %p = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @A, i64 0, i64 %k, i64 %x, i64 %i
  %v = load double, ptr %p, align 8
  %w = fadd double %v, 1.0
  store double %w, ptr %p, align 8
  %xn = add i64 %x, 1
  %xe = icmp eq i64 %xn, 7
  br i1 %xe, label %li, label %bx
li:
  %in = add i64 %i, 1
  %ie = icmp eq i64 %in, 7
  br i1 %ie, label %prea, label %hi
prea:
  br label %ha
ha:
  %a = phi i64 [0, %prea], [%an, %la]
  br label %hb
hb:
  %b = phi i64 [0, %ha], [%bn, %lb]
  br label %bc
bc:
  %c = phi i64 [0, %hb], [%cn, %bc]
  %q = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @B, i64 0, i64 %a, i64 %c, i64 %b
  %cv = load double, ptr %q, align 8
  %cw = fadd double %cv, 1.0
  store double %cw, ptr %q, align 8
  %cn = add i64 %c, 1
  %ce = icmp eq i64 %cn, 7
  br i1 %ce, label %lb, label %bc
lb:
  %bn = add i64 %b, 1
  %be = icmp eq i64 %bn, 7
  br i1 %be, label %la, label %hb
la:
  %an = add i64 %a, 1
  %ae = icmp eq i64 %an, 7
  br i1 %ae, label %lk, label %ha
lk:
  %kn = add i64 %k, 1
  %ke = icmp eq i64 %kn, 7
  br i1 %ke, label %exit, label %hk
exit:
  ret void
}

; The hi/bx pair is swapped: the former inner header %bx now heads the loop
; directly nested in %hk, and the former outer header %hi now heads the
; innermost loop. The deeper ha/hb/bc triple, which has no eligible pair within
; the cap, is untouched.
; TOO-DEEP-LOOPS-LABEL: Loop info for function 'too_deep_reaches_shallow_pair':
; TOO-DEEP-LOOPS:         Loop at depth 1 containing: %hk<header>
; TOO-DEEP-LOOPS-NEXT:      Loop at depth 2 containing: %bx<header>
; TOO-DEEP-LOOPS-NEXT:        Loop at depth 3 containing: %hi<header>
; TOO-DEEP-LOOPS-NEXT:      Loop at depth 2 containing: %ha<header>
; TOO-DEEP-LOOPS-NEXT:        Loop at depth 3 containing: %hb<header>
; TOO-DEEP-LOOPS-NEXT:          Loop at depth 4 containing: %bc<header>
;
; Disabling fallback keeps both subtrees in their original order.
; FALLBACK-DISABLED-LOOPS-LABEL: Loop info for function 'too_deep_reaches_shallow_pair':
; FALLBACK-DISABLED-LOOPS:         Loop at depth 1 containing: %hk<header>
; FALLBACK-DISABLED-LOOPS-NEXT:      Loop at depth 2 containing: %hi<header>
; FALLBACK-DISABLED-LOOPS-NEXT:        Loop at depth 3 containing: %bx<header>
; FALLBACK-DISABLED-LOOPS-NEXT:      Loop at depth 2 containing: %ha<header>
; FALLBACK-DISABLED-LOOPS-NEXT:        Loop at depth 3 containing: %hb<header>
; FALLBACK-DISABLED-LOOPS-NEXT:          Loop at depth 4 containing: %bc<header>

;-------------------------------------------------------------------------------
; Four eligible sibling-chain pairs under a common ancestor k (unused in any
; index, so its surrounding dependence direction is unknown, as in
; budget_exhausted_all_fail above), at four distinct depths: p3 (depth 3), p4
; (depth 4, one wrapper level deep), p5 (depth 5, two wrapper levels deep), and
; p6 (depth 6, three wrapper levels deep). With the budget at 2, retention
; keeps only the Budget+1 deepest candidates while scanning (p6, p5, p4),
; evicting the shallowest surplus (p3) outright the moment a 4th candidate (p6)
; arrives and pushes the retained count 2 over budget; this exercises
; Candidates.pop_back(). Of the retained three, only the two deepest (p6, p5)
; are attempted, deepest first; p4 is kept solely to anchor the
; budget-exhaustion remark.
;-------------------------------------------------------------------------------
define void @eviction_prunes_shallowest_surplus() !dbg !5 {
entry:
  br label %k.header

k.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %k.latch ]
  br label %p3.i.header

p3.i.header:
  %i3 = phi i64 [ 0, %k.header ], [ %i3.next, %p3.i.latch ]
  br label %p3.j.body, !dbg !6

p3.j.body:
  %j3 = phi i64 [ 1, %p3.i.header ], [ %j3.next, %p3.j.body ]
  %j3m1 = sub i64 %j3, 1
  %p3.ld = getelementptr inbounds [8 x [8 x double]], ptr @DA, i64 0, i64 %j3, i64 %i3
  %p3.v = load double, ptr %p3.ld, align 8
  %p3.nv = fadd double %p3.v, 1.000000e+00
  %p3.st = getelementptr inbounds [8 x [8 x double]], ptr @DA, i64 0, i64 %j3m1, i64 %i3
  store double %p3.nv, ptr %p3.st, align 8
  %j3.next = add i64 %j3, 1
  %j3.ec = icmp eq i64 %j3.next, 8
  br i1 %j3.ec, label %p3.i.latch, label %p3.j.body

p3.i.latch:
  %i3.next = add i64 %i3, 1
  %i3.ec = icmp eq i64 %i3.next, 7
  br i1 %i3.ec, label %m4.header, label %p3.i.header

m4.header:
  %m4 = phi i64 [ 0, %p3.i.latch ], [ %m4.next, %m4.latch ]
  br label %p4.i.header

p4.i.header:
  %i4 = phi i64 [ 0, %m4.header ], [ %i4.next, %p4.i.latch ]
  br label %p4.j.body, !dbg !7

p4.j.body:
  %j4 = phi i64 [ 1, %p4.i.header ], [ %j4.next, %p4.j.body ]
  %j4m1 = sub i64 %j4, 1
  %p4.ld = getelementptr inbounds [8 x [8 x double]], ptr @DB, i64 0, i64 %j4, i64 %i4
  %p4.v = load double, ptr %p4.ld, align 8
  %p4.nv = fadd double %p4.v, 1.000000e+00
  %p4.st = getelementptr inbounds [8 x [8 x double]], ptr @DB, i64 0, i64 %j4m1, i64 %i4
  store double %p4.nv, ptr %p4.st, align 8
  %j4.next = add i64 %j4, 1
  %j4.ec = icmp eq i64 %j4.next, 8
  br i1 %j4.ec, label %p4.i.latch, label %p4.j.body

p4.i.latch:
  %i4.next = add i64 %i4, 1
  %i4.ec = icmp eq i64 %i4.next, 7
  br i1 %i4.ec, label %m4.latch, label %p4.i.header

m4.latch:
  %m4.next = add i64 %m4, 1
  %m4.ec = icmp eq i64 %m4.next, 4
  br i1 %m4.ec, label %m5a.header, label %m4.header

m5a.header:
  %m5a = phi i64 [ 0, %m4.latch ], [ %m5a.next, %m5a.latch ]
  br label %m5b.header

m5b.header:
  %m5b = phi i64 [ 0, %m5a.header ], [ %m5b.next, %m5b.latch ]
  br label %p5.i.header

p5.i.header:
  %i5 = phi i64 [ 0, %m5b.header ], [ %i5.next, %p5.i.latch ]
  br label %p5.j.body, !dbg !8

p5.j.body:
  %j5 = phi i64 [ 1, %p5.i.header ], [ %j5.next, %p5.j.body ]
  %j5m1 = sub i64 %j5, 1
  %p5.ld = getelementptr inbounds [8 x [8 x double]], ptr @DC, i64 0, i64 %j5, i64 %i5
  %p5.v = load double, ptr %p5.ld, align 8
  %p5.nv = fadd double %p5.v, 1.000000e+00
  %p5.st = getelementptr inbounds [8 x [8 x double]], ptr @DC, i64 0, i64 %j5m1, i64 %i5
  store double %p5.nv, ptr %p5.st, align 8
  %j5.next = add i64 %j5, 1
  %j5.ec = icmp eq i64 %j5.next, 8
  br i1 %j5.ec, label %p5.i.latch, label %p5.j.body

p5.i.latch:
  %i5.next = add i64 %i5, 1
  %i5.ec = icmp eq i64 %i5.next, 7
  br i1 %i5.ec, label %m5b.latch, label %p5.i.header

m5b.latch:
  %m5b.next = add i64 %m5b, 1
  %m5b.ec = icmp eq i64 %m5b.next, 4
  br i1 %m5b.ec, label %m5a.latch, label %m5b.header

m5a.latch:
  %m5a.next = add i64 %m5a, 1
  %m5a.ec = icmp eq i64 %m5a.next, 4
  br i1 %m5a.ec, label %m6a.header, label %m5a.header

m6a.header:
  %m6a = phi i64 [ 0, %m5a.latch ], [ %m6a.next, %m6a.latch ]
  br label %m6b.header

m6b.header:
  %m6b = phi i64 [ 0, %m6a.header ], [ %m6b.next, %m6b.latch ]
  br label %m6c.header

m6c.header:
  %m6c = phi i64 [ 0, %m6b.header ], [ %m6c.next, %m6c.latch ]
  br label %p6.i.header

p6.i.header:
  %i6 = phi i64 [ 0, %m6c.header ], [ %i6.next, %p6.i.latch ]
  br label %p6.j.body, !dbg !9

p6.j.body:
  %j6 = phi i64 [ 1, %p6.i.header ], [ %j6.next, %p6.j.body ]
  %j6m1 = sub i64 %j6, 1
  %p6.ld = getelementptr inbounds [8 x [8 x double]], ptr @DD, i64 0, i64 %j6, i64 %i6
  %p6.v = load double, ptr %p6.ld, align 8
  %p6.nv = fadd double %p6.v, 1.000000e+00
  %p6.st = getelementptr inbounds [8 x [8 x double]], ptr @DD, i64 0, i64 %j6m1, i64 %i6
  store double %p6.nv, ptr %p6.st, align 8
  %j6.next = add i64 %j6, 1
  %j6.ec = icmp eq i64 %j6.next, 8
  br i1 %j6.ec, label %p6.i.latch, label %p6.j.body

p6.i.latch:
  %i6.next = add i64 %i6, 1
  %i6.ec = icmp eq i64 %i6.next, 7
  br i1 %i6.ec, label %m6c.latch, label %p6.i.header

m6c.latch:
  %m6c.next = add i64 %m6c, 1
  %m6c.ec = icmp eq i64 %m6c.next, 4
  br i1 %m6c.ec, label %m6b.latch, label %m6c.header

m6b.latch:
  %m6b.next = add i64 %m6b, 1
  %m6b.ec = icmp eq i64 %m6b.next, 4
  br i1 %m6b.ec, label %m6a.latch, label %m6b.header

m6a.latch:
  %m6a.next = add i64 %m6a, 1
  %m6a.ec = icmp eq i64 %m6a.next, 4
  br i1 %m6a.ec, label %k.latch, label %m6a.header

k.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 4
  br i1 %k.ec, label %exit, label %k.header

exit:
  ret void
}

; Eviction leaves every candidate pair unswapped and in its original sibling
; order: the evicted depth-3 pair (p3), the unattempted-anchor depth-4 pair
; (p4), and the two attempted, still-rejected depth-5 and depth-6 pairs (p5,
; p6) all keep their original outer-then-inner header order.
; EVICT-LOOPS-LABEL: Loop info for function 'eviction_prunes_shallowest_surplus':
; EVICT-LOOPS:         Loop at depth 1 containing: %k.header<header>
; EVICT-LOOPS-NEXT:      Loop at depth 2 containing: %p3.i.header<header>
; EVICT-LOOPS-NEXT:        Loop at depth 3 containing: %p3.j.body<header>
; EVICT-LOOPS-NEXT:      Loop at depth 2 containing: %m4.header<header>
; EVICT-LOOPS-NEXT:        Loop at depth 3 containing: %p4.i.header<header>
; EVICT-LOOPS-NEXT:          Loop at depth 4 containing: %p4.j.body<header>
; EVICT-LOOPS-NEXT:      Loop at depth 2 containing: %m5a.header<header>
; EVICT-LOOPS-NEXT:        Loop at depth 3 containing: %m5b.header<header>
; EVICT-LOOPS-NEXT:          Loop at depth 4 containing: %p5.i.header<header>
; EVICT-LOOPS-NEXT:            Loop at depth 5 containing: %p5.j.body<header>
; EVICT-LOOPS-NEXT:      Loop at depth 2 containing: %m6a.header<header>
; EVICT-LOOPS-NEXT:        Loop at depth 3 containing: %m6b.header<header>
; EVICT-LOOPS-NEXT:          Loop at depth 4 containing: %m6c.header<header>
; EVICT-LOOPS-NEXT:            Loop at depth 5 containing: %p6.i.header<header>
; EVICT-LOOPS-NEXT:              Loop at depth 6 containing: %p6.j.body<header>

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "llvm", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "inner-subnest-budget.ll", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !2)
!5 = distinct !DISubprogram(name: "eviction_prunes_shallowest_surplus", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DILocation(line: 30, column: 1, scope: !5)
!7 = !DILocation(line: 40, column: 1, scope: !5)
!8 = !DILocation(line: 50, column: 1, scope: !5)
!9 = !DILocation(line: 60, column: 1, scope: !5)
!10 = distinct !DISubprogram(name: "budget_exhausted_all_fail", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!11 = !DILocation(line: 70, column: 1, scope: !10)
!12 = !DILocation(line: 80, column: 1, scope: !10)
!13 = !DILocation(line: 90, column: 1, scope: !10)
