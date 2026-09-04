; Behavior test for inner-subnest candidate formation in LoopInterchange.
;
; This file is entirely hand-maintained. Do NOT run update_test_checks.py on it.
;
; Background:
;   LoopInterchangePass::run consumes LoopNest::getLoops(), which is a
;   *breadth-first* walk over every descendant loop (siblings included).
;   The pass (1) uses LoopNest::getNestDepth() for the depth policy instead of
;   that descendant count, so a shallow sibling-rich nest is not misclassified
;   as too deep; and (2) when the whole breadth-first nest is unsuitable for
;   the standard multi-swap path -- it is non-linear, a loop in its linear
;   chain is uncomputable or unsupported, or it exceeds the depth policy --
;   enumerates direct parent/leaf-child edges (a parent with a single child
;   loop that is itself a leaf) and performs at most one sound, profitable
;   interchange through the inner-subnest fallback. The standard multi-swap
;   path and its behavior are unchanged.
;
; A fixed leading dimension of 1335 doubles reproduces canonical SWIM's
; 10,680-byte inner stride (1335 * 8). The two `admitted_*` functions below are
; standalone, admissible 2-deep nests that the *standard* path interchanges
; under default profitability; they establish that the shared candidate pair is
; legal and profitable "once admitted". Five positive fixtures embed that
; same shape inside an enclosing structure the standard path rejects; the
; fallback reaches and interchanges the eligible pair there, where the
; unmodified pass left the nest alone. The lasting negatives stay in their
; original order.
;
; The applied/analysis/missed remark stream (YAML) is the primary decision and
; diagnostic oracle. The IR run additionally pins that surviving structure
; (address expressions, reassociated reductions, sibling traffic) is preserved
; and that the analysis verifiers pass. Because the "Interchanged" remark is emitted
; *before* the transform runs and carries no loop identity, a third run pins the
; produced loop hierarchy with `print<loops>`: after a fallback interchange the
; former inner header block heads the new outer loop and the former outer header
; block heads the new inner loop, which distinguishes a real swap from an
; unswapped nest and, for sibling-rich parents, proves the parent's sibling
; program order is preserved (see the LOOPS lines).
;
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -S 2>&1 | FileCheck %s --check-prefix=IR
;
; Post-transform loop hierarchy. LoopInterchange preserves LoopAnalysis, so the
; following print<loops> reflects the pass's incrementally-updated loop tree
; (headers, nesting, and sibling order), not a fresh rebuild.
; RUN: opt < %s -passes='loop(loop-interchange),print<loops>' -cache-line-size=64 \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=LOOPS
;
; Full, function-associated remark log (Passed / Missed / Analysis). The
; --implicit-check-not proves the depth misclassification is fixed: with the
; true nesting depth used for the policy, no function is rejected for depth.
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -pass-remarks=loop-interchange -pass-remarks-missed=loop-interchange \
; RUN:     -pass-remarks-output=%t -disable-output
; RUN: FileCheck %s --check-prefix=YAML --input-file=%t \
; RUN:     --implicit-check-not=UnsupportedLoopNestDepth
;
; The switch gates fallback processing, not the true-depth policy. With
; fallback disabled, this non-linear nest is no longer rejected for depth and
; is not interchanged, so it emits an empty remark file. `test -e` makes the
; emptiness check non-vacuous.
; RUN: llvm-extract -S -func=bfs_loop_count_is_not_depth %s \
; RUN:     -o %t.disabled-nonlinear
; RUN: opt < %t.disabled-nonlinear -passes=loop-interchange \
; RUN:     -cache-line-size=64 \
; RUN:     -loop-interchange-enable-inner-subnest-fallback=false \
; RUN:     -pass-remarks-output=%t.disabled-nonlinear.yaml -disable-output
; RUN: test -e %t.disabled-nonlinear.yaml
; RUN: test ! -s %t.disabled-nonlinear.yaml
; RUN: opt < %t.disabled-nonlinear \
; RUN:     -passes='loop(loop-interchange),print<loops>' -cache-line-size=64 \
; RUN:     -loop-interchange-enable-inner-subnest-fallback=false \
; RUN:     -disable-output 2>&1 | \
; RUN:     FileCheck %s --check-prefix=ORIGINAL-NONLINEAR-LOOPS
;
; RUN: llvm-extract -S -func=uncomputable_sibling_does_not_block %s \
; RUN:     -o %t.disabled-uncomputable
; RUN: opt < %t.disabled-uncomputable -passes=loop-interchange \
; RUN:     -cache-line-size=64 \
; RUN:     -loop-interchange-enable-inner-subnest-fallback=false \
; RUN:     -pass-remarks-output=%t.disabled-uncomputable.yaml -disable-output
; RUN: test -e %t.disabled-uncomputable.yaml
; RUN: test ! -s %t.disabled-uncomputable.yaml
; RUN: opt < %t.disabled-uncomputable \
; RUN:     -passes='loop(loop-interchange),print<loops>' -cache-line-size=64 \
; RUN:     -loop-interchange-enable-inner-subnest-fallback=false \
; RUN:     -disable-output 2>&1 | \
; RUN:     FileCheck %s --check-prefix=DISABLED-UNCOMPUTABLE-LOOPS

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

;-------------------------------------------------------------------------------
; Expected remark log (function/module order). See per-function notes.
; The two admitted_* controls interchange through the standard path; five
; positive fixtures interchange through the inner-subnest fallback. Lasting
; negatives are rejected by profitability, strict reductions, an unsupported
; pair, or failure to form an eligible leaf candidate.
;-------------------------------------------------------------------------------
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        admitted_1335_pair_one_reduction
; YAML:      --- !Passed
; YAML:      Name:            Interchanged
; YAML:      Function:        admitted_1335_pair_one_reduction
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        admitted_1335_pair_three_reductions
; YAML:      --- !Passed
; YAML:      Name:            Interchanged
; YAML:      Function:        admitted_1335_pair_three_reductions
; bfs_loop_count_is_not_depth is no longer rejected for depth (true depth 3)
; and is interchanged through the fallback.
; YAML:      --- !Passed
; YAML:      Name:            Interchanged
; YAML:      Function:        bfs_loop_count_is_not_depth
; The non-linear uncomputable sibling reaches fallback directly. The linear
; uncomputable ancestor exercises the whole-nest computability rejection. Both
; interchange their computable inner pair without an analysis remark.
; YAML:      --- !Passed
; YAML:      Name:            Interchanged
; YAML:      Function:        uncomputable_sibling_does_not_block
; YAML:      --- !Passed
; YAML:      Name:            Interchanged
; YAML:      Function:        uncomputable_ancestor_partition
; two_candidate_pairs_one_fallback has two eligible pairs; exactly one (the
; deterministic first) is interchanged through the fallback.
; YAML:      --- !Passed
; YAML:      Name:            Interchanged
; YAML:      Function:        two_candidate_pairs_one_fallback
; YAML-NOT:  Function:        two_candidate_pairs_one_fallback
; partly_exact_reduction is a plain admissible 2-deep nest on the standard path;
; its strict fadd is rejected exactly as before.
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        partly_exact_reduction
; YAML:      --- !Missed
; YAML:      Name:            UnsupportedPHIOuter
; YAML:      Function:        partly_exact_reduction
; The dynamic leading dimension is reached by the fallback but declined by
; default profitability.
; YAML:      --- !Missed
; YAML:      Name:            InterchangeNotProfitable
; YAML:      Function:        dynamic_leading_dimension_subnest
; The all-exact and partly-exact subnests are reached by the fallback but their
; strict recurrences are rejected.
; YAML:      --- !Missed
; YAML:      Name:            UnsupportedPHIOuter
; YAML:      Function:        all_exact_reduction_subnest
; YAML:      --- !Missed
; YAML:      Name:            UnsupportedPHIOuter
; YAML:      Function:        partly_exact_reduction_subnest
; non_leaf_candidate_subnest forms no eligible candidate (its only single-child
; edge has a non-leaf inner loop), so the fallback leaves it unchanged.
; distinct_depth_deepest_first has two eligible pairs at different depths; the
; deeper pair (dB, inner depth 4) is selected deepest-first and interchanged, and
; the shallower pair (sA) is left untouched (at most one transform).
; YAML:      --- !Passed
; YAML:      Name:            Interchanged
; YAML:      Function:        distinct_depth_deepest_first
; YAML-NOT:  Function:        distinct_depth_deepest_first
; unsupported_pair_backedge_subnest's non-linear list reaches fallback
; directly. Its i/j pair has two exiting blocks (no unique exit), so pair-local
; computability rejects it as unsupported without an analysis remark.
; YAML:      --- !Missed
; YAML:      Name:            FallbackUnsupportedPair
; YAML:      Function:        unsupported_pair_backedge_subnest
;

;-------------------------------------------------------------------------------
; Positive controls: the shared fixed-1335 reduction pair, presented as a plain
; admissible 2-deep nest, is interchanged by the standard path under default
; profitability. These prove the pair is legal + profitable "once admitted", so
; every fallback case below is reached and interchanged (or rejected) purely on
; its own merits. The Passed records are selection/invocation oracles; the LOOPS
; checks pin the resulting hierarchy.
;-------------------------------------------------------------------------------

; double sum = 0; for i: for j: sum += A[j][i];   (inner j strides 1335 doubles)
define void @admitted_1335_pair_one_reduction(ptr %A, ptr %R) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %sum.i = phi double [ 0.000000e+00, %entry ], [ %sum.i.lcssa, %outer.latch ]
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %sum.j = phi double [ %sum.i, %outer.header ], [ %sum.j.next, %inner.header ]
  %idx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  %a = load double, ptr %idx, align 8
  %sum.j.next = fadd reassoc double %sum.j, %a
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %outer.latch, label %inner.header

outer.latch:
  %sum.i.lcssa = phi double [ %sum.j.next, %inner.header ]
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %exit, label %outer.header

exit:
  %sum.res = phi double [ %sum.i.lcssa, %outer.latch ]
  store double %sum.res, ptr %R, align 8
  ret void
}

; Standard-path control: this admissible 2-deep nest is interchanged by the
; multi-swap path, not the fallback. After the swap the former inner header
; %inner.header heads the new outer loop and the former outer header
; %outer.header heads the new inner loop (a real swap, not just a surviving
; nest). This also guards that the standard multi-swap path is unchanged by the
; inner-subnest fallback.
; LOOPS-LABEL: Loop info for function 'admitted_1335_pair_one_reduction':
; LOOPS:         Loop at depth 1 containing: %inner.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %outer.header<header>

; Three independent reassociated reductions, matching the SWIM checksum shape.
define void @admitted_1335_pair_three_reductions(ptr %A, ptr %B, ptr %C, ptr %R) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %sumA.i = phi double [ 0.000000e+00, %entry ], [ %sumA.i.lcssa, %outer.latch ]
  %sumB.i = phi double [ 0.000000e+00, %entry ], [ %sumB.i.lcssa, %outer.latch ]
  %sumC.i = phi double [ 0.000000e+00, %entry ], [ %sumC.i.lcssa, %outer.latch ]
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %sumA.j = phi double [ %sumA.i, %outer.header ], [ %sumA.j.next, %inner.header ]
  %sumB.j = phi double [ %sumB.i, %outer.header ], [ %sumB.j.next, %inner.header ]
  %sumC.j = phi double [ %sumC.i, %outer.header ], [ %sumC.j.next, %inner.header ]
  %idxA = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  %a = load double, ptr %idxA, align 8
  %sumA.j.next = fadd reassoc double %sumA.j, %a
  %idxB = getelementptr inbounds [1335 x double], ptr %B, i64 %j, i64 %i
  %b = load double, ptr %idxB, align 8
  %sumB.j.next = fadd reassoc double %sumB.j, %b
  %idxC = getelementptr inbounds [1335 x double], ptr %C, i64 %j, i64 %i
  %c = load double, ptr %idxC, align 8
  %sumC.j.next = fadd reassoc double %sumC.j, %c
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %outer.latch, label %inner.header

outer.latch:
  %sumA.i.lcssa = phi double [ %sumA.j.next, %inner.header ]
  %sumB.i.lcssa = phi double [ %sumB.j.next, %inner.header ]
  %sumC.i.lcssa = phi double [ %sumC.j.next, %inner.header ]
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %exit, label %outer.header

exit:
  %sumA.res = phi double [ %sumA.i.lcssa, %outer.latch ]
  %sumB.res = phi double [ %sumB.i.lcssa, %outer.latch ]
  %sumC.res = phi double [ %sumC.i.lcssa, %outer.latch ]
  %rB = getelementptr inbounds double, ptr %R, i64 1
  %rC = getelementptr inbounds double, ptr %R, i64 2
  store double %sumA.res, ptr %R, align 8
  store double %sumB.res, ptr %rB, align 8
  store double %sumC.res, ptr %rC, align 8
  ret void
}

; Standard-path control (three reductions). Same swap oracle as the one-reduction
; control: former inner header %inner.header heads the new outer loop.
; LOOPS-LABEL: Loop info for function 'admitted_1335_pair_three_reductions':
; LOOPS:         Loop at depth 1 containing: %inner.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %outer.header<header>

;-------------------------------------------------------------------------------
; (1) Sibling-rich, genuinely shallow nest (true depth 3) whose breadth-first
; descendant count is 12 (top + pairX.outer + pairX.inner + sib1..sib9). The
; unmodified pass reported that flat count as an unsupported "depth" and never
; considered the eligible fixed-1335 three-reduction pairX.outer/pairX.inner;
; the depth policy now uses the true nesting depth, so the nest is no longer
; rejected for depth. Its non-linear list routes directly to the fallback,
; which interchanges that pair.
;-------------------------------------------------------------------------------
define void @bfs_loop_count_is_not_depth(ptr %A, ptr %B, ptr %C, ptr %R) {
entry:
  br label %top.header

top.header:
  %t = phi i64 [ 0, %entry ], [ %t.next, %top.latch ]
  br label %pairX.outer.header

pairX.outer.header:
  %i = phi i64 [ 0, %top.header ], [ %i.next, %pairX.outer.latch ]
  %sumA.i = phi double [ 0.000000e+00, %top.header ], [ %sumA.i.lcssa, %pairX.outer.latch ]
  %sumB.i = phi double [ 0.000000e+00, %top.header ], [ %sumB.i.lcssa, %pairX.outer.latch ]
  %sumC.i = phi double [ 0.000000e+00, %top.header ], [ %sumC.i.lcssa, %pairX.outer.latch ]
  br label %pairX.inner

pairX.inner:
  %j = phi i64 [ 0, %pairX.outer.header ], [ %j.next, %pairX.inner ]
  %sumA.j = phi double [ %sumA.i, %pairX.outer.header ], [ %sumA.j.next, %pairX.inner ]
  %sumB.j = phi double [ %sumB.i, %pairX.outer.header ], [ %sumB.j.next, %pairX.inner ]
  %sumC.j = phi double [ %sumC.i, %pairX.outer.header ], [ %sumC.j.next, %pairX.inner ]
  %idxA = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  %a = load double, ptr %idxA, align 8
  %sumA.j.next = fadd reassoc double %sumA.j, %a
  %idxB = getelementptr inbounds [1335 x double], ptr %B, i64 %j, i64 %i
  %b = load double, ptr %idxB, align 8
  %sumB.j.next = fadd reassoc double %sumB.j, %b
  %idxC = getelementptr inbounds [1335 x double], ptr %C, i64 %j, i64 %i
  %c = load double, ptr %idxC, align 8
  %sumC.j.next = fadd reassoc double %sumC.j, %c
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %pairX.outer.latch, label %pairX.inner

pairX.outer.latch:
  %sumA.i.lcssa = phi double [ %sumA.j.next, %pairX.inner ]
  %sumB.i.lcssa = phi double [ %sumB.j.next, %pairX.inner ]
  %sumC.i.lcssa = phi double [ %sumC.j.next, %pairX.inner ]
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %pairX.exit, label %pairX.outer.header

pairX.exit:
  %sumA.live = phi double [ %sumA.i.lcssa, %pairX.outer.latch ]
  %sumB.live = phi double [ %sumB.i.lcssa, %pairX.outer.latch ]
  %sumC.live = phi double [ %sumC.i.lcssa, %pairX.outer.latch ]
  %rB = getelementptr inbounds double, ptr %R, i64 1
  %rC = getelementptr inbounds double, ptr %R, i64 2
  store double %sumA.live, ptr %R, align 8
  store double %sumB.live, ptr %rB, align 8
  store double %sumC.live, ptr %rC, align 8
  br label %sib1.header

sib1.header:
  %s1 = phi i64 [ 0, %pairX.exit ], [ %s1.next, %sib1.header ]
  %s1.next = add i64 %s1, 1
  %s1.ec = icmp eq i64 %s1.next, 4
  br i1 %s1.ec, label %sib1.exit, label %sib1.header

sib1.exit:
  br label %sib2.header

sib2.header:
  %s2 = phi i64 [ 0, %sib1.exit ], [ %s2.next, %sib2.header ]
  %s2.next = add i64 %s2, 1
  %s2.ec = icmp eq i64 %s2.next, 4
  br i1 %s2.ec, label %sib2.exit, label %sib2.header

sib2.exit:
  br label %sib3.header

sib3.header:
  %s3 = phi i64 [ 0, %sib2.exit ], [ %s3.next, %sib3.header ]
  %s3.next = add i64 %s3, 1
  %s3.ec = icmp eq i64 %s3.next, 4
  br i1 %s3.ec, label %sib3.exit, label %sib3.header

sib3.exit:
  br label %sib4.header

sib4.header:
  %s4 = phi i64 [ 0, %sib3.exit ], [ %s4.next, %sib4.header ]
  %s4.next = add i64 %s4, 1
  %s4.ec = icmp eq i64 %s4.next, 4
  br i1 %s4.ec, label %sib4.exit, label %sib4.header

sib4.exit:
  br label %sib5.header

sib5.header:
  %s5 = phi i64 [ 0, %sib4.exit ], [ %s5.next, %sib5.header ]
  %s5.next = add i64 %s5, 1
  %s5.ec = icmp eq i64 %s5.next, 4
  br i1 %s5.ec, label %sib5.exit, label %sib5.header

sib5.exit:
  br label %sib6.header

sib6.header:
  %s6 = phi i64 [ 0, %sib5.exit ], [ %s6.next, %sib6.header ]
  %s6.next = add i64 %s6, 1
  %s6.ec = icmp eq i64 %s6.next, 4
  br i1 %s6.ec, label %sib6.exit, label %sib6.header

sib6.exit:
  br label %sib7.header

sib7.header:
  %s7 = phi i64 [ 0, %sib6.exit ], [ %s7.next, %sib7.header ]
  %s7.next = add i64 %s7, 1
  %s7.ec = icmp eq i64 %s7.next, 4
  br i1 %s7.ec, label %sib7.exit, label %sib7.header

sib7.exit:
  br label %sib8.header

sib8.header:
  %s8 = phi i64 [ 0, %sib7.exit ], [ %s8.next, %sib8.header ]
  %s8.next = add i64 %s8, 1
  %s8.ec = icmp eq i64 %s8.next, 4
  br i1 %s8.ec, label %sib8.exit, label %sib8.header

sib8.exit:
  br label %sib9.header

sib9.header:
  %s9 = phi i64 [ 0, %sib8.exit ], [ %s9.next, %sib9.header ]
  %s9.next = add i64 %s9, 1
  %s9.ec = icmp eq i64 %s9.next, 4
  br i1 %s9.ec, label %sib9.exit, label %sib9.header

sib9.exit:
  br label %top.latch

top.latch:
  %t.next = add i64 %t, 1
  %t.ec = icmp eq i64 %t.next, 4
  br i1 %t.ec, label %exit, label %top.header

exit:
  ret void
}

; The pairX candidate pair is interchanged through the fallback. The enclosing
; top loop and the nine sibling loops are untouched. The three address
; expressions and reassociated reductions are preserved, and the three checksum
; results are still stored to %R (observable live-outs). The LOOPS checks below
; prove the swap and preserved sibling program order.
; IR-LABEL: define void @bfs_loop_count_is_not_depth(
; IR-DAG:     %t = phi i64 [ 0, %entry ], [ %t.next, %top.latch ]
; IR-DAG:     %idxA = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
; IR-DAG:     %idxB = getelementptr inbounds [1335 x double], ptr %B, i64 %j, i64 %i
; IR-DAG:     %idxC = getelementptr inbounds [1335 x double], ptr %C, i64 %j, i64 %i
; IR-DAG:     %sumA.j.next = fadd reassoc double %sumA.j, %a
; IR-DAG:     %sumB.j.next = fadd reassoc double %sumB.j, %b
; IR-DAG:     %sumC.j.next = fadd reassoc double %sumC.j, %c
; IR-DAG:     store double %{{.*}}, ptr %R, align 8
; IR-DAG:     store double %{{.*}}, ptr %rB, align 8
; IR-DAG:     store double %{{.*}}, ptr %rC, align 8
; IR-DAG:     %s1 = phi i64 [ 0, %{{.*}} ], [ %s1.next, %sib1.header ]
; IR-DAG:     %s9 = phi i64 [ 0, %{{.*}} ], [ %s9.next, %sib9.header ]
; IR-DAG:     %t.next = add i64 %t, 1

; Swap + sibling-order oracle. The interchanged pair's new outer loop is headed
; by the former inner header %pairX.inner and its new inner loop by the former
; outer header %pairX.outer.header (proving a real swap). Crucially, that pair
; remains the *first* of top's ten children, ahead of the nine untouched sibling
; loops sib1..sib9 in program order: a LoopInfo update that removed the old child
; and appended the replacement would push it behind %sib9.header instead.
; LOOPS-LABEL: Loop info for function 'bfs_loop_count_is_not_depth':
; LOOPS:         Loop at depth 1 containing: %top.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %pairX.inner<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %pairX.outer.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %sib1.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %sib2.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %sib3.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %sib4.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %sib5.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %sib6.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %sib7.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %sib8.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %sib9.header<header>
;
; With fallback disabled, the profitable pair keeps its original order.
; ORIGINAL-NONLINEAR-LOOPS-LABEL: Loop info for function 'bfs_loop_count_is_not_depth':
; ORIGINAL-NONLINEAR-LOOPS:         Loop at depth 1 containing: %top.header<header>
; ORIGINAL-NONLINEAR-LOOPS-NEXT:      Loop at depth 2 containing: %pairX.outer.header<header>
; ORIGINAL-NONLINEAR-LOOPS-NEXT:        Loop at depth 3 containing: %pairX.inner<header>
; ORIGINAL-NONLINEAR-LOOPS-NEXT:      Loop at depth 2 containing: %sib1.header<header>
; ORIGINAL-NONLINEAR-LOOPS-NEXT:      Loop at depth 2 containing: %sib2.header<header>
; ORIGINAL-NONLINEAR-LOOPS-NEXT:      Loop at depth 2 containing: %sib3.header<header>
; ORIGINAL-NONLINEAR-LOOPS-NEXT:      Loop at depth 2 containing: %sib4.header<header>
; ORIGINAL-NONLINEAR-LOOPS-NEXT:      Loop at depth 2 containing: %sib5.header<header>
; ORIGINAL-NONLINEAR-LOOPS-NEXT:      Loop at depth 2 containing: %sib6.header<header>
; ORIGINAL-NONLINEAR-LOOPS-NEXT:      Loop at depth 2 containing: %sib7.header<header>
; ORIGINAL-NONLINEAR-LOOPS-NEXT:      Loop at depth 2 containing: %sib8.header<header>
; ORIGINAL-NONLINEAR-LOOPS-NEXT:      Loop at depth 2 containing: %sib9.header<header>

;-------------------------------------------------------------------------------
; (2) A single SCEV-uncomputable sibling loop (data-dependent exit) sits beside
; a separate, computable, profitable fixed-1335 pair under a common ancestor.
; The non-linear list routes directly to fallback, which forms the computable
; pair and interchanges it without querying the unrelated sibling's trip count.
;-------------------------------------------------------------------------------
define void @uncomputable_sibling_does_not_block(ptr %A, ptr %U, ptr %R) {
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
  br i1 %i.ec, label %pair.exit, label %pair.outer.header

pair.exit:
  %sum.live = phi double [ %sum.i.lcssa, %pair.outer.latch ]
  store double %sum.live, ptr %R, align 8
  br label %usib.header

usib.header:
  %s = phi i64 [ 0, %pair.exit ], [ %s.next, %usib.header ]
  %sp = getelementptr inbounds double, ptr %U, i64 %s
  %sv = load double, ptr %sp, align 8
  %sc = fcmp oeq double %sv, 0.000000e+00
  %s.next = add i64 %s, 1
  br i1 %sc, label %usib.exit, label %usib.header

usib.exit:
  br label %anc.latch

anc.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 4
  br i1 %k.ec, label %exit, label %anc.header

exit:
  ret void
}

; The computable pair is interchanged through the fallback; the uncomputable
; sibling is untouched and still exits on a loaded value. The address
; expression and reassociated reduction are preserved and the result is stored
; to %R.
; IR-LABEL: define void @uncomputable_sibling_does_not_block(
; IR-DAG:     %idx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
; IR-DAG:     %sum.j.next = fadd reassoc double %sum.j, %a
; IR-DAG:     store double %{{.*}}, ptr %R, align 8
; IR-DAG:     %sv = load double, ptr %sp, align 8
; IR-DAG:     %sc = fcmp oeq double %sv, 0.000000e+00

; Swap + sibling-order oracle: the interchanged pair's new outer loop is headed
; by the former inner header %pair.inner (new inner headed by %pair.outer.header),
; and it stays ahead of the untouched uncomputable sibling loop %usib.header.
; LOOPS-LABEL: Loop info for function 'uncomputable_sibling_does_not_block':
; LOOPS:         Loop at depth 1 containing: %anc.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %pair.inner<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %pair.outer.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %usib.header<header>
;
; This non-linear nest routes to fallback before any whole-nest analysis
; remark. With fallback disabled it emits no remark, and the computable pair
; keeps its original order. The uncomputable sibling is incidental here; the
; linear uncomputable ancestor below exercises computability rejection.
; DISABLED-UNCOMPUTABLE-LOOPS-LABEL: Loop info for function 'uncomputable_sibling_does_not_block':
; DISABLED-UNCOMPUTABLE-LOOPS:         Loop at depth 1 containing: %anc.header<header>
; DISABLED-UNCOMPUTABLE-LOOPS-NEXT:      Loop at depth 2 containing: %pair.outer.header<header>
; DISABLED-UNCOMPUTABLE-LOOPS-NEXT:        Loop at depth 3 containing: %pair.inner<header>
; DISABLED-UNCOMPUTABLE-LOOPS-NEXT:      Loop at depth 2 containing: %usib.header<header>

;-------------------------------------------------------------------------------
; (3) A lower, computable fixed-1335 pair beneath an uncomputable *true
; ancestor* (data-dependent latch). The chain is linear, but isComputableLoopNest
; rejects it because of the ancestor, so the standard path never considers the
; lower pair; the fallback partitions that pair off and interchanges it.
;-------------------------------------------------------------------------------
define void @uncomputable_ancestor_partition(ptr %A, ptr %U, ptr %R) {
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
  %kp = getelementptr inbounds double, ptr %U, i64 %k
  %kv = load double, ptr %kp, align 8
  %kc = fcmp oeq double %kv, 0.000000e+00
  %k.next = add i64 %k, 1
  br i1 %kc, label %exit, label %anc.header

exit:
  ret void
}

; The lower computable pair is interchanged through the fallback even though its
; true ancestor is uncomputable; the ancestor loop and its data-dependent latch
; are untouched. The address expression and reassociated reduction are
; preserved and the result is stored to %R.
; IR-LABEL: define void @uncomputable_ancestor_partition(
; IR-DAG:     %k = phi i64 [ 0, %entry ], [ %k.next, %anc.latch ]
; IR-DAG:     %idx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
; IR-DAG:     %sum.j.next = fadd reassoc double %sum.j, %a
; IR-DAG:     store double %{{.*}}, ptr %R, align 8
; IR-DAG:     %kv = load double, ptr %kp, align 8
; IR-DAG:     %kc = fcmp oeq double %kv, 0.000000e+00

; Swap oracle (3-level nest). The uncomputable ancestor %anc.header stays the
; outermost loop; beneath it the interchanged pair's new outer loop is headed by
; the former inner header %pair.inner and the new inner loop by the former outer
; header %pair.outer.header.
; LOOPS-LABEL: Loop info for function 'uncomputable_ancestor_partition':
; LOOPS:         Loop at depth 1 containing: %anc.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %pair.inner<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %pair.outer.header<header>

;-------------------------------------------------------------------------------
; (4) Two eligible direct single-child pairs (pairA, pairB) share one ancestor,
; making the breadth-first list non-linear. It routes directly to fallback,
; whose one-transform policy interchanges exactly one pair.
;-------------------------------------------------------------------------------
define void @two_candidate_pairs_one_fallback(ptr %A, ptr %B, ptr %R) {
entry:
  br label %anc.header

anc.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %anc.latch ]
  br label %pairA.outer.header

pairA.outer.header:
  %iA = phi i64 [ 0, %anc.header ], [ %iA.next, %pairA.outer.latch ]
  %sumA.i = phi double [ 0.000000e+00, %anc.header ], [ %sumA.i.lcssa, %pairA.outer.latch ]
  br label %pairA.inner

pairA.inner:
  %jA = phi i64 [ 0, %pairA.outer.header ], [ %jA.next, %pairA.inner ]
  %sumA.j = phi double [ %sumA.i, %pairA.outer.header ], [ %sumA.j.next, %pairA.inner ]
  %idxA = getelementptr inbounds [1335 x double], ptr %A, i64 %jA, i64 %iA
  %a = load double, ptr %idxA, align 8
  %sumA.j.next = fadd reassoc double %sumA.j, %a
  %jA.next = add i64 %jA, 1
  %jA.ec = icmp eq i64 %jA.next, 1335
  br i1 %jA.ec, label %pairA.outer.latch, label %pairA.inner

pairA.outer.latch:
  %sumA.i.lcssa = phi double [ %sumA.j.next, %pairA.inner ]
  %iA.next = add i64 %iA, 1
  %iA.ec = icmp eq i64 %iA.next, 1335
  br i1 %iA.ec, label %pairA.exit, label %pairA.outer.header

pairA.exit:
  %sumA.live = phi double [ %sumA.i.lcssa, %pairA.outer.latch ]
  store double %sumA.live, ptr %R, align 8
  br label %pairB.outer.header

pairB.outer.header:
  %iB = phi i64 [ 0, %pairA.exit ], [ %iB.next, %pairB.outer.latch ]
  %sumB.i = phi double [ 0.000000e+00, %pairA.exit ], [ %sumB.i.lcssa, %pairB.outer.latch ]
  br label %pairB.inner

pairB.inner:
  %jB = phi i64 [ 0, %pairB.outer.header ], [ %jB.next, %pairB.inner ]
  %sumB.j = phi double [ %sumB.i, %pairB.outer.header ], [ %sumB.j.next, %pairB.inner ]
  %idxB = getelementptr inbounds [1335 x double], ptr %B, i64 %jB, i64 %iB
  %b = load double, ptr %idxB, align 8
  %sumB.j.next = fadd reassoc double %sumB.j, %b
  %jB.next = add i64 %jB, 1
  %jB.ec = icmp eq i64 %jB.next, 1335
  br i1 %jB.ec, label %pairB.outer.latch, label %pairB.inner

pairB.outer.latch:
  %sumB.i.lcssa = phi double [ %sumB.j.next, %pairB.inner ]
  %iB.next = add i64 %iB, 1
  %iB.ec = icmp eq i64 %iB.next, 1335
  br i1 %iB.ec, label %pairB.exit, label %pairB.outer.header

pairB.exit:
  %sumB.live = phi double [ %sumB.i.lcssa, %pairB.outer.latch ]
  %rB = getelementptr inbounds double, ptr %R, i64 1
  store double %sumB.live, ptr %rB, align 8
  br label %anc.latch

anc.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 4
  br i1 %k.ec, label %exit, label %anc.header

exit:
  ret void
}

; Exactly one pair is interchanged through the fallback. The deterministic first
; candidate (pairA) is interchanged; the second (pairB) is left in its original
; inner-reduction order, proving at most one interchange per invocation. Both
; address expressions and reassociated reductions are preserved and both
; results are stored.
; IR-LABEL: define void @two_candidate_pairs_one_fallback(
; IR-DAG:     %idxA = getelementptr inbounds [1335 x double], ptr %A, i64 %jA, i64 %iA
; IR-DAG:     %sumA.j.next = fadd reassoc double %sumA.j, %a
; IR-DAG:     store double %{{.*}}, ptr %R, align 8
; IR-DAG:     %sumB.j = phi double [ %sumB.i, %pairB.outer.header ], [ %sumB.j.next, %pairB.inner ]
; IR-DAG:     %idxB = getelementptr inbounds [1335 x double], ptr %B, i64 %jB, i64 %iB
; IR-DAG:     %sumB.j.next = fadd reassoc double %sumB.j, %b
; IR-DAG:     store double %sumB.live, ptr %rB, align 8

; One-transform + deepest/first-selection oracle. pairA (the deterministic first
; candidate) is swapped -- its new outer loop is headed by the former inner
; header %pairA.inner -- while pairB is left unswapped: %pairB.outer.header still
; heads its outer loop with %pairB.inner nested inside. pairA also stays the
; first of anc's two children (sibling order preserved).
; LOOPS-LABEL: Loop info for function 'two_candidate_pairs_one_fallback':
; LOOPS:         Loop at depth 1 containing: %anc.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %pairA.inner<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %pairA.outer.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %pairB.outer.header<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %pairB.inner<header>

;-------------------------------------------------------------------------------
; (5a) Lasting negative: a partly-exact reduction set. sumA is reassociable but
; sumB is a strict fadd, so even though this is a plain admissible 2-deep nest
; the pass refuses it (UnsupportedPHIOuter), and the fallback must refuse it
; too -- reassociation is required on every reordered recurrence.
;-------------------------------------------------------------------------------
define void @partly_exact_reduction(ptr %A, ptr %B, ptr %R) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %sumA.i = phi double [ 0.000000e+00, %entry ], [ %sumA.i.lcssa, %outer.latch ]
  %sumB.i = phi double [ 0.000000e+00, %entry ], [ %sumB.i.lcssa, %outer.latch ]
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %sumA.j = phi double [ %sumA.i, %outer.header ], [ %sumA.j.next, %inner.header ]
  %sumB.j = phi double [ %sumB.i, %outer.header ], [ %sumB.j.next, %inner.header ]
  %idxA = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  %a = load double, ptr %idxA, align 8
  %sumA.j.next = fadd reassoc double %sumA.j, %a
  %idxB = getelementptr inbounds [1335 x double], ptr %B, i64 %j, i64 %i
  %b = load double, ptr %idxB, align 8
  %sumB.j.next = fadd double %sumB.j, %b
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %outer.latch, label %inner.header

outer.latch:
  %sumA.i.lcssa = phi double [ %sumA.j.next, %inner.header ]
  %sumB.i.lcssa = phi double [ %sumB.j.next, %inner.header ]
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %exit, label %outer.header

exit:
  %sumA.res = phi double [ %sumA.i.lcssa, %outer.latch ]
  %sumB.res = phi double [ %sumB.i.lcssa, %outer.latch ]
  %rB = getelementptr inbounds double, ptr %R, i64 1
  store double %sumA.res, ptr %R, align 8
  store double %sumB.res, ptr %rB, align 8
  ret void
}

; Both reductions keep their original order; note the strict (non-reassoc) fadd.
; IR-LABEL: define void @partly_exact_reduction(
; IR:         %sumA.j = phi double [ %sumA.i, %outer.header ], [ %sumA.j.next, %inner.header ]
; IR:         %sumB.j = phi double [ %sumB.i, %outer.header ], [ %sumB.j.next, %inner.header ]
; IR:         %sumA.j.next = fadd reassoc double %sumA.j, %a
; IR:         %sumB.j.next = fadd double %sumB.j, %b
; IR:         %sumA.i.lcssa = phi double [ %sumA.j.next, %inner.header ]
; IR:         %sumB.i.lcssa = phi double [ %sumB.j.next, %inner.header ]
; IR:         %sumA.res = phi double [ %sumA.i.lcssa, %outer.latch ]
; IR:         %sumB.res = phi double [ %sumB.i.lcssa, %outer.latch ]
; IR:         %rB = getelementptr inbounds double, ptr %R, i64 1
; IR:         store double %sumA.res, ptr %R, align 8
; IR:         store double %sumB.res, ptr %rB, align 8

;-------------------------------------------------------------------------------
; (5b) Lasting negative: a dynamic leading dimension (A[j*n + i]). Here it is
; embedded beside a sibling loop, so the non-linear nest routes directly to
; fallback. Default profitability declines the dynamic stride, so it stays out
; of scope and the nest is not transformed.
;-------------------------------------------------------------------------------
define void @dynamic_leading_dimension_subnest(ptr %A, i64 %n, ptr %U, ptr %R) {
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
  %rowoff = mul i64 %j, %n
  %off = add i64 %rowoff, %i
  %idx = getelementptr inbounds double, ptr %A, i64 %off
  %a = load double, ptr %idx, align 8
  %sum.j.next = fadd reassoc double %sum.j, %a
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %pair.outer.latch, label %pair.inner

pair.outer.latch:
  %sum.i.lcssa = phi double [ %sum.j.next, %pair.inner ]
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %pair.exit, label %pair.outer.header

pair.exit:
  %sum.live = phi double [ %sum.i.lcssa, %pair.outer.latch ]
  store double %sum.live, ptr %R, align 8
  br label %sib.header

sib.header:
  %s = phi i64 [ 0, %pair.exit ], [ %s.next, %sib.header ]
  %sp = getelementptr inbounds double, ptr %U, i64 %s
  %sv = load double, ptr %sp, align 8
  %sd = fadd reassoc double %sv, 1.000000e+00
  store double %sd, ptr %sp, align 8
  %s.next = add i64 %s, 1
  %s.ec = icmp eq i64 %s.next, 4
  br i1 %s.ec, label %sib.exit, label %sib.header

sib.exit:
  br label %anc.latch

anc.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 4
  br i1 %k.ec, label %exit, label %anc.header

exit:
  ret void
}

; The dynamic-stride address and reduction order are preserved.
; IR-LABEL: define void @dynamic_leading_dimension_subnest(
; IR:         %sum.j = phi double [ %sum.i, %pair.outer.header ], [ %sum.j.next, %pair.inner ]
; IR:         %rowoff = mul i64 %j, %n
; IR:         %off = add i64 %rowoff, %i
; IR:         %idx = getelementptr inbounds double, ptr %A, i64 %off
; IR:         %sum.j.next = fadd reassoc double %sum.j, %a
; IR:         %sum.i.lcssa = phi double [ %sum.j.next, %pair.inner ]
; IR:         %sum.live = phi double [ %sum.i.lcssa, %pair.outer.latch ]
; IR:         store double %sum.live, ptr %R, align 8
; IR:         %sd = fadd reassoc double %sv, 1.000000e+00

;-------------------------------------------------------------------------------
; (5c) Lasting negative inside a fallback-triggering shape: an all-exact (strict,
; non-reassoc) reduction pair nested under an ancestor k beside a sibling loop, so
; the non-linear flat list routes directly to fallback. The fallback reaches
; this direct i/j pair but declines it -- reordering a strict fadd changes the
; result, so reassoc is required on every reordered recurrence. The result is
; stored (real live-out).
;-------------------------------------------------------------------------------
define void @all_exact_reduction_subnest(ptr %A, ptr %U, ptr %R) {
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
  %sum.j.next = fadd double %sum.j, %a
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %pair.outer.latch, label %pair.inner

pair.outer.latch:
  %sum.i.lcssa = phi double [ %sum.j.next, %pair.inner ]
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %pair.exit, label %pair.outer.header

pair.exit:
  %sum.live = phi double [ %sum.i.lcssa, %pair.outer.latch ]
  store double %sum.live, ptr %R, align 8
  br label %sib.header

sib.header:
  %s = phi i64 [ 0, %pair.exit ], [ %s.next, %sib.header ]
  %sp = getelementptr inbounds double, ptr %U, i64 %s
  %sv = load double, ptr %sp, align 8
  %sd = fadd reassoc double %sv, 1.000000e+00
  store double %sd, ptr %sp, align 8
  %s.next = add i64 %s, 1
  %s.ec = icmp eq i64 %s.next, 4
  br i1 %s.ec, label %sib.exit, label %sib.header

sib.exit:
  br label %anc.latch

anc.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 4
  br i1 %k.ec, label %exit, label %anc.header

exit:
  ret void
}

; The strict (non-reassoc) reduction cycle and its address stay in original
; order; the observable store and the sibling traffic remain in place.
; IR-LABEL: define void @all_exact_reduction_subnest(
; IR:         %sum.i = phi double [ 0.000000e+00, %anc.header ], [ %sum.i.lcssa, %pair.outer.latch ]
; IR:         %sum.j = phi double [ %sum.i, %pair.outer.header ], [ %sum.j.next, %pair.inner ]
; IR:         %idx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
; IR:         %sum.j.next = fadd double %sum.j, %a
; IR:         %sum.i.lcssa = phi double [ %sum.j.next, %pair.inner ]
; IR:         %sum.live = phi double [ %sum.i.lcssa, %pair.outer.latch ]
; IR:         store double %sum.live, ptr %R, align 8

; Lasting-negative structural oracle: the strict reduction is declined, so the
; pair is NOT swapped -- the former outer header %pair.outer.header still heads
; the depth-2 loop with %pair.inner nested at depth 3, and the sibling remains.
; LOOPS-LABEL: Loop info for function 'all_exact_reduction_subnest':
; LOOPS:         Loop at depth 1 containing: %anc.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %pair.outer.header<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %pair.inner<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %sib.header<header>

;-------------------------------------------------------------------------------
; (5d) Lasting negative inside the same fallback-triggering shape: a partly-exact
; reduction pair (sumA reassociable, sumB a strict fadd) nested under ancestor k
; beside a sibling, so the non-linear nest routes directly to fallback. The
; fallback reaches the i/j pair but declines it -- every reordered recurrence
; must be reassociable and sumB is not. Both results are stored.
;-------------------------------------------------------------------------------
define void @partly_exact_reduction_subnest(ptr %A, ptr %B, ptr %U, ptr %R) {
entry:
  br label %anc.header

anc.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %anc.latch ]
  br label %pair.outer.header

pair.outer.header:
  %i = phi i64 [ 0, %anc.header ], [ %i.next, %pair.outer.latch ]
  %sumA.i = phi double [ 0.000000e+00, %anc.header ], [ %sumA.i.lcssa, %pair.outer.latch ]
  %sumB.i = phi double [ 0.000000e+00, %anc.header ], [ %sumB.i.lcssa, %pair.outer.latch ]
  br label %pair.inner

pair.inner:
  %j = phi i64 [ 0, %pair.outer.header ], [ %j.next, %pair.inner ]
  %sumA.j = phi double [ %sumA.i, %pair.outer.header ], [ %sumA.j.next, %pair.inner ]
  %sumB.j = phi double [ %sumB.i, %pair.outer.header ], [ %sumB.j.next, %pair.inner ]
  %idxA = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  %a = load double, ptr %idxA, align 8
  %sumA.j.next = fadd reassoc double %sumA.j, %a
  %idxB = getelementptr inbounds [1335 x double], ptr %B, i64 %j, i64 %i
  %b = load double, ptr %idxB, align 8
  %sumB.j.next = fadd double %sumB.j, %b
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %pair.outer.latch, label %pair.inner

pair.outer.latch:
  %sumA.i.lcssa = phi double [ %sumA.j.next, %pair.inner ]
  %sumB.i.lcssa = phi double [ %sumB.j.next, %pair.inner ]
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %pair.exit, label %pair.outer.header

pair.exit:
  %sumA.live = phi double [ %sumA.i.lcssa, %pair.outer.latch ]
  %sumB.live = phi double [ %sumB.i.lcssa, %pair.outer.latch ]
  %rB = getelementptr inbounds double, ptr %R, i64 1
  store double %sumA.live, ptr %R, align 8
  store double %sumB.live, ptr %rB, align 8
  br label %sib.header

sib.header:
  %s = phi i64 [ 0, %pair.exit ], [ %s.next, %sib.header ]
  %sp = getelementptr inbounds double, ptr %U, i64 %s
  %sv = load double, ptr %sp, align 8
  %sd = fadd reassoc double %sv, 1.000000e+00
  store double %sd, ptr %sp, align 8
  %s.next = add i64 %s, 1
  %s.ec = icmp eq i64 %s.next, 4
  br i1 %s.ec, label %sib.exit, label %sib.header

sib.exit:
  br label %anc.latch

anc.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 4
  br i1 %k.ec, label %exit, label %anc.header

exit:
  ret void
}

; sumA is reassociable and sumB is a strict fadd; both keep their original order,
; both results are stored, and the sibling traffic remains.
; IR-LABEL: define void @partly_exact_reduction_subnest(
; IR:         %sumA.j.next = fadd reassoc double %sumA.j, %a
; IR:         %sumB.j.next = fadd double %sumB.j, %b
; IR:         %sumA.i.lcssa = phi double [ %sumA.j.next, %pair.inner ]
; IR:         %sumB.i.lcssa = phi double [ %sumB.j.next, %pair.inner ]
; IR:         %sumA.live = phi double [ %sumA.i.lcssa, %pair.outer.latch ]
; IR:         %sumB.live = phi double [ %sumB.i.lcssa, %pair.outer.latch ]
; IR:         %rB = getelementptr inbounds double, ptr %R, i64 1
; IR:         store double %sumA.live, ptr %R, align 8
; IR:         store double %sumB.live, ptr %rB, align 8

;-------------------------------------------------------------------------------
; (5e) Lasting negative: an unsupported (non-leaf) candidate structure. The
; eligible-looking i/j pair has an inner loop j that is itself NOT a leaf -- it
; encloses two sibling loops m and n -- so j has two children and the flat list
; is non-linear and routes directly to fallback. The fallback only selects a
; leaf inner loop, so this non-leaf candidate is skipped and the nest is
; unchanged. The candidate store is observable.
;-------------------------------------------------------------------------------
define void @non_leaf_candidate_subnest(ptr %A) {
entry:
  br label %i.header

i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %i.latch ]
  br label %j.header

j.header:
  %j = phi i64 [ 0, %i.header ], [ %j.next, %j.latch ]
  %idx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  store double 1.000000e+00, ptr %idx, align 8
  br label %m.header

m.header:
  %m = phi i64 [ 0, %j.header ], [ %m.next, %m.header ]
  %m.next = add i64 %m, 1
  %m.ec = icmp eq i64 %m.next, 4
  br i1 %m.ec, label %m.exit, label %m.header

m.exit:
  br label %n.header

n.header:
  %n = phi i64 [ 0, %m.exit ], [ %n.next, %n.header ]
  %n.next = add i64 %n, 1
  %n.ec = icmp eq i64 %n.next, 4
  br i1 %n.ec, label %j.latch, label %n.header

j.latch:
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %i.latch, label %j.header

i.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %exit, label %i.header

exit:
  ret void
}

; The i/j candidate keeps its original order and address, and the inner loop j
; still encloses the two sibling loops m and n (its non-leaf body). The store is
; the observable result.
; IR-LABEL: define void @non_leaf_candidate_subnest(
; IR:         %i = phi i64 [ 0, %entry ], [ %i.next, %i.latch ]
; IR:         %j = phi i64 [ 0, %i.header ], [ %j.next, %j.latch ]
; IR:         %idx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
; IR:         store double 1.000000e+00, ptr %idx, align 8
; IR:         %m = phi i64 [ 0, %j.header ], [ %m.next, %m.header ]
; IR:         %n = phi i64 [ 0, %m.exit ], [ %n.next, %n.header ]

; Lasting-negative structural oracle: no eligible candidate forms (j is non-leaf),
; so nothing is swapped -- i outer, j middle, with the two leaf loops m and n.
; LOOPS-LABEL: Loop info for function 'non_leaf_candidate_subnest':
; LOOPS:         Loop at depth 1 containing: %i.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %j.header<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %m.header<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %n.header<header>

;-------------------------------------------------------------------------------
; (6) Distinct-depth deepest-first selection. Two eligible fixed-1335 reduction
; pairs are siblings under a common top loop (so the breadth-first list is
; non-linear): a shallow pair sA (inner depth 3) and, one level deeper, a pair dB
; (inner depth 4). The fallback tries the deepest candidate first, so dB is the
; one interchanged; sA -- equally eligible -- is left untouched because at most
; one interchange happens per invocation.
;-------------------------------------------------------------------------------
define void @distinct_depth_deepest_first(ptr %A, ptr %B, ptr %R) {
entry:
  br label %top.header

top.header:
  %t = phi i64 [ 0, %entry ], [ %t.next, %top.latch ]
  br label %sA.outer.header

sA.outer.header:
  %ia = phi i64 [ 0, %top.header ], [ %ia.next, %sA.outer.latch ]
  %sumA.i = phi double [ 0.000000e+00, %top.header ], [ %sumA.i.lcssa, %sA.outer.latch ]
  br label %sA.inner

sA.inner:
  %ja = phi i64 [ 0, %sA.outer.header ], [ %ja.next, %sA.inner ]
  %sumA.j = phi double [ %sumA.i, %sA.outer.header ], [ %sumA.j.next, %sA.inner ]
  %idxA = getelementptr inbounds [1335 x double], ptr %A, i64 %ja, i64 %ia
  %a = load double, ptr %idxA, align 8
  %sumA.j.next = fadd reassoc double %sumA.j, %a
  %ja.next = add i64 %ja, 1
  %ja.ec = icmp eq i64 %ja.next, 1335
  br i1 %ja.ec, label %sA.outer.latch, label %sA.inner

sA.outer.latch:
  %sumA.i.lcssa = phi double [ %sumA.j.next, %sA.inner ]
  %ia.next = add i64 %ia, 1
  %ia.ec = icmp eq i64 %ia.next, 1335
  br i1 %ia.ec, label %sA.exit, label %sA.outer.header

sA.exit:
  %sumA.live = phi double [ %sumA.i.lcssa, %sA.outer.latch ]
  store double %sumA.live, ptr %R, align 8
  br label %mid.header

mid.header:
  %m = phi i64 [ 0, %sA.exit ], [ %m.next, %mid.latch ]
  br label %dB.outer.header

dB.outer.header:
  %ib = phi i64 [ 0, %mid.header ], [ %ib.next, %dB.outer.latch ]
  %sumB.i = phi double [ 0.000000e+00, %mid.header ], [ %sumB.i.lcssa, %dB.outer.latch ]
  br label %dB.inner

dB.inner:
  %jb = phi i64 [ 0, %dB.outer.header ], [ %jb.next, %dB.inner ]
  %sumB.j = phi double [ %sumB.i, %dB.outer.header ], [ %sumB.j.next, %dB.inner ]
  %idxB = getelementptr inbounds [1335 x double], ptr %B, i64 %jb, i64 %ib
  %b = load double, ptr %idxB, align 8
  %sumB.j.next = fadd reassoc double %sumB.j, %b
  %jb.next = add i64 %jb, 1
  %jb.ec = icmp eq i64 %jb.next, 1335
  br i1 %jb.ec, label %dB.outer.latch, label %dB.inner

dB.outer.latch:
  %sumB.i.lcssa = phi double [ %sumB.j.next, %dB.inner ]
  %ib.next = add i64 %ib, 1
  %ib.ec = icmp eq i64 %ib.next, 1335
  br i1 %ib.ec, label %dB.exit, label %dB.outer.header

dB.exit:
  %sumB.live = phi double [ %sumB.i.lcssa, %dB.outer.latch ]
  %rB = getelementptr inbounds double, ptr %R, i64 1
  store double %sumB.live, ptr %rB, align 8
  br label %mid.latch

mid.latch:
  %m.next = add i64 %m, 1
  %m.ec = icmp eq i64 %m.next, 4
  br i1 %m.ec, label %top.latch, label %mid.header

top.latch:
  %t.next = add i64 %t, 1
  %t.ec = icmp eq i64 %t.next, 4
  br i1 %t.ec, label %exit, label %top.header

exit:
  ret void
}

; The deeper pair dB is interchanged; the shallower pair sA keeps its original
; inner-reduction order (its inner PHI still seeds from %sA.outer.header). Both
; address expressions, both reassociated reductions, and both live-out stores
; survive.
; IR-LABEL: define void @distinct_depth_deepest_first(
; IR-DAG:     %sumA.j = phi double [ %sumA.i, %sA.outer.header ], [ %sumA.j.next, %sA.inner ]
; IR-DAG:     %idxA = getelementptr inbounds [1335 x double], ptr %A, i64 %ja, i64 %ia
; IR-DAG:     %sumA.j.next = fadd reassoc double %sumA.j, %a
; IR-DAG:     %idxB = getelementptr inbounds [1335 x double], ptr %B, i64 %jb, i64 %ib
; IR-DAG:     %sumB.j.next = fadd reassoc double %sumB.j, %b
; IR-DAG:     store double %sumA.live, ptr %R, align 8
; IR-DAG:     store double %sumB.live, ptr %rB, align 8

; Deepest-first oracle: only the deeper pair dB is swapped (its new outer loop is
; headed by the former inner header %dB.inner, at depth 3, above the former outer
; header %dB.outer.header at depth 4), while the shallower pair sA is left
; unswapped (%sA.outer.header still heads its depth-2 loop above %sA.inner).
; LOOPS-LABEL: Loop info for function 'distinct_depth_deepest_first':
; LOOPS:         Loop at depth 1 containing: %top.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %sA.outer.header<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %sA.inner<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %mid.header<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %dB.inner<header>
; LOOPS-NEXT:          Loop at depth 4 containing: %dB.outer.header<header>

;-------------------------------------------------------------------------------
; (7) The selected pair's own structure is unsupported: the candidate inner loop
; has two exiting blocks (an early exit at j==500 and the normal exit), so it has
; no unique exit and is not a computable inner pair. It sits beside a sibling loop
; so the flat list is non-linear and reaches the fallback, which rejects this one
; candidate with FallbackUnsupportedPair and leaves the nest unchanged.
;-------------------------------------------------------------------------------
define void @unsupported_pair_backedge_subnest(ptr %A, ptr %U) {
entry:
  br label %anc.header

anc.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %anc.latch ]
  br label %pair.outer.header

pair.outer.header:
  %i = phi i64 [ 0, %anc.header ], [ %i.next, %pair.outer.latch ]
  br label %pair.inner.header

pair.inner.header:
  %j = phi i64 [ 0, %pair.outer.header ], [ %j.next, %pair.inner.latch ]
  %idx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  store double 1.000000e+00, ptr %idx, align 8
  %early = icmp eq i64 %j, 500
  br i1 %early, label %pair.exit.early, label %pair.inner.latch

pair.inner.latch:
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %pair.exit.normal, label %pair.inner.header

pair.exit.early:
  br label %pair.outer.latch

pair.exit.normal:
  br label %pair.outer.latch

pair.outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %sib.preheader, label %pair.outer.header

sib.preheader:
  br label %sib.header

sib.header:
  %s = phi i64 [ 0, %sib.preheader ], [ %s.next, %sib.header ]
  %sp = getelementptr inbounds double, ptr %U, i64 %s
  %sv = load double, ptr %sp, align 8
  %s.next = add i64 %s, 1
  %s.ec = icmp eq i64 %s.next, 4
  br i1 %s.ec, label %sib.exit, label %sib.header

sib.exit:
  br label %anc.latch

anc.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 4
  br i1 %k.ec, label %exit, label %anc.header

exit:
  ret void
}

; The pair keeps its original order, its two-exit inner loop, and the store; the
; sibling is untouched. Nothing is interchanged (FallbackUnsupportedPair).
; IR-LABEL: define void @unsupported_pair_backedge_subnest(
; IR:         %i = phi i64 [ 0, %anc.header ], [ %i.next, %pair.outer.latch ]
; IR:         %j = phi i64 [ 0, %pair.outer.header ], [ %j.next, %pair.inner.latch ]
; IR:         %idx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
; IR:         br i1 %early, label %pair.exit.early, label %pair.inner.latch
; LOOPS-LABEL: Loop info for function 'unsupported_pair_backedge_subnest':
; LOOPS:         Loop at depth 1 containing: %anc.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %pair.outer.header<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %pair.inner.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %sib.header<header>
