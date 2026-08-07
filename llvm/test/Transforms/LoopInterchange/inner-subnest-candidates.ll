; Precommit test for inner-subnest candidate formation in LoopInterchange.
;
; This file is entirely hand-maintained. Do NOT run update_test_checks.py on
; it: the CHECK lines deliberately pin the *current* no-transform behavior of
; the unmodified LoopInterchange pass so that the follow-on
; candidate-formation implementation can show a reviewable diff.
;
; Background:
;   LoopInterchangePass::run consumes LoopNest::getLoops(), which is a
;   *breadth-first* walk over every descendant loop (siblings included). The
;   pass then (1) rejects when that flat list is longer than
;   MaxLoopNestDepth (=10), (2) rejects when any member has an uncomputable
;   backedge / non-unique exit, and (3) in LoopInterchange::run(LoopNest&)
;   rejects when the flat list is not a single linear chain. In each of those
;   cases the pass bails *before* it ever reaches an otherwise eligible,
;   profitable adjacent parent/child pair, so no interchange happens today.
;
; A fixed leading dimension of 1335 doubles gives a 10,680-byte inner stride
; (1335 * 8), a cache-hostile column-major access typical of shallow-water
; stencil benchmarks. The two `admitted_*` functions below are standalone,
; admissible 2-deep nests that the *current* pass already interchanges under
; default profitability; they establish that the shared candidate pair is legal
; and profitable "once admitted". Every other function embeds that same shape
; (or a deliberately-permanent negative) inside an enclosing structure that the
; current pass rejects, and pins that the pair is left in its original order.
;
; This precommit only asserts behavior observable on the unmodified pass: the
; breadth-first depth remark, the applied/analysis/missed remarks, and the
; actual unswapped IR. Direct-edge candidate selection, ancestor-column
; dependence handling and sibling exclusion are oracles for the follow-on
; candidate-formation implementation and are intentionally NOT asserted here.
;
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -S 2>&1 | FileCheck %s --check-prefix=IR
;
; Full, function-associated remark log (Passed / Missed / Analysis).
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -pass-remarks=loop-interchange -pass-remarks-missed=loop-interchange \
; RUN:     -pass-remarks-output=%t -disable-output
; RUN: FileCheck %s --check-prefix=YAML --input-file=%t
;
; Raising the depth cap past the breadth-first count removes the (false) depth
; rejection for the shallow, sibling-rich nest, proving the count -- not a real
; depth-3 problem -- triggered it. Under the raised cap bfs_loop_count_is_not_depth
; clears the depth gate and reaches dependence analysis (its own function-
; associated !Analysis Dependence record) instead of UnsupportedLoopNestDepth; it
; still does not interchange (its next blocker is non-linearity), which the
; follow-on candidate-formation implementation addresses. Proven from the
; function-associated YAML remark stream, not a shared stderr string that
; another function could satisfy.
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-max-loop-nest-depth=32 \
; RUN:     -pass-remarks-output=%t.raised -disable-output
; RUN: FileCheck %s --check-prefix=RAISED --input-file=%t.raised \
; RUN:     --implicit-check-not=UnsupportedLoopNestDepth

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

;-------------------------------------------------------------------------------
; Expected current remark log (function/module order). See per-function notes.
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
; YAML:      --- !Missed
; YAML:      Name:            UnsupportedLoopNestDepth
; YAML:      Function:        bfs_loop_count_is_not_depth
; The two uncomputable-neighbour nests bail before the analysis remark, so they
; emit nothing at all between the depth miss and the two-candidate analysis.
; YAML-NOT:  Function:        uncomputable_sibling_does_not_block
; YAML-NOT:  Function:        uncomputable_ancestor_partition
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        two_candidate_pairs_one_fallback
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        partly_exact_reduction
; YAML:      --- !Missed
; YAML:      Name:            UnsupportedPHIOuter
; YAML:      Function:        partly_exact_reduction
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        dynamic_leading_dimension_subnest
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        all_exact_reduction_subnest
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        partly_exact_reduction_subnest
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        non_leaf_candidate_subnest

; The two admitted controls still interchange (two Interchanged records); the very
; next analysis record is bfs_loop_count_is_not_depth, proving it now clears the
; raised depth gate and reaches dependence analysis rather than being rejected as
; too deep. --implicit-check-not proves no UnsupportedLoopNestDepth is emitted.
; RAISED:      Name:            Interchanged
; RAISED:      Name:            Interchanged
; RAISED:      Name:            Dependence
; RAISED-NEXT: Function:        bfs_loop_count_is_not_depth

;-------------------------------------------------------------------------------
; Positive controls: the shared fixed-1335 reduction pair, presented as a plain
; admissible 2-deep nest, is interchanged by the current pass under default
; profitability. These prove the pair is legal + profitable "once admitted", so
; every blocked case below fails only because of its enclosing structure.
; Their transformed IR is not pinned here (that belongs to the follow-on
; behavior commit's before/after); the Passed remark above is the oracle.
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

; Three independent reassociated reductions over three arrays, the shape of a
; multi-array checksum loop.
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

;-------------------------------------------------------------------------------
; (1) Sibling-rich, genuinely shallow nest (true depth 3) whose breadth-first
; descendant count is 12 (top + pairX.outer + pairX.inner + sib1..sib9). The
; current pass reports the flat count as an unsupported "depth" and never
; considers the eligible fixed-1335 three-reduction pairX.outer/pairX.inner.
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

; The pairX reduction cycle and its address expression are unchanged, and the
; inner reduction PHIs still take their initial value from %pairX.outer.header
; (interchange would rewire these incoming edges). The pair remains nested
; between %top.header and %top.latch, with the sibling chain intact.
; IR-LABEL: define void @bfs_loop_count_is_not_depth(
; IR:         %t = phi i64 [ 0, %entry ], [ %t.next, %top.latch ]
; IR:         %sumA.i = phi double [ 0.000000e+00, %top.header ], [ %sumA.i.lcssa, %pairX.outer.latch ]
; IR:         %sumC.i = phi double [ 0.000000e+00, %top.header ], [ %sumC.i.lcssa, %pairX.outer.latch ]
; IR:         %j = phi i64 [ 0, %pairX.outer.header ], [ %j.next, %pairX.inner ]
; IR:         %sumA.j = phi double [ %sumA.i, %pairX.outer.header ], [ %sumA.j.next, %pairX.inner ]
; IR:         %sumC.j = phi double [ %sumC.i, %pairX.outer.header ], [ %sumC.j.next, %pairX.inner ]
; IR:         %idxA = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
; IR:         %sumA.j.next = fadd reassoc double %sumA.j, %a
; IR:         %sumC.j.next = fadd reassoc double %sumC.j, %c
; IR:         %sumA.i.lcssa = phi double [ %sumA.j.next, %pairX.inner ]
; IR:         %sumA.live = phi double [ %sumA.i.lcssa, %pairX.outer.latch ]
; IR:         store double %sumA.live, ptr %R, align 8
; IR:         %s1 = phi i64 [ 0, %pairX.exit ], [ %s1.next, %sib1.header ]
; IR:         %s9 = phi i64 [ 0, %sib8.exit ], [ %s9.next, %sib9.header ]
; IR:         %t.next = add i64 %t, 1

;-------------------------------------------------------------------------------
; (2) A single SCEV-uncomputable sibling loop (data-dependent exit) sits beside
; a separate, computable, profitable fixed-1335 pair under a common ancestor.
; isComputableLoopNest rejects the whole flat list, so the pair is not reached
; and no analysis remark is emitted.
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

; The computable pair keeps its original order; the uncomputable sibling still
; exits on a loaded value.
; IR-LABEL: define void @uncomputable_sibling_does_not_block(
; IR:         %sum.i = phi double [ 0.000000e+00, %anc.header ], [ %sum.i.lcssa, %pair.outer.latch ]
; IR:         %sum.j = phi double [ %sum.i, %pair.outer.header ], [ %sum.j.next, %pair.inner ]
; IR:         %idx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
; IR:         %sum.j.next = fadd reassoc double %sum.j, %a
; IR:         %sum.i.lcssa = phi double [ %sum.j.next, %pair.inner ]
; IR:         %sum.live = phi double [ %sum.i.lcssa, %pair.outer.latch ]
; IR:         store double %sum.live, ptr %R, align 8
; IR:         %sv = load double, ptr %sp, align 8
; IR:         %sc = fcmp oeq double %sv, 0.000000e+00

;-------------------------------------------------------------------------------
; (3) A lower, computable fixed-1335 pair beneath an uncomputable *true
; ancestor* (data-dependent latch). The chain is linear, but isComputableLoopNest
; rejects it because of the ancestor, so the lower pair is never partitioned off
; and considered.
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

; The pair is untouched and still nested under the uncomputable ancestor.
; IR-LABEL: define void @uncomputable_ancestor_partition(
; IR:         %k = phi i64 [ 0, %entry ], [ %k.next, %anc.latch ]
; IR:         %sum.i = phi double [ 0.000000e+00, %anc.header ], [ %sum.i.lcssa, %pair.outer.latch ]
; IR:         %sum.j = phi double [ %sum.i, %pair.outer.header ], [ %sum.j.next, %pair.inner ]
; IR:         %idx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
; IR:         %sum.j.next = fadd reassoc double %sum.j, %a
; IR:         %sum.i.lcssa = phi double [ %sum.j.next, %pair.inner ]
; IR:         %sum.live = phi double [ %sum.i.lcssa, %pair.outer.latch ]
; IR:         store double %sum.live, ptr %R, align 8
; IR:         %kv = load double, ptr %kp, align 8
; IR:         %kc = fcmp oeq double %kv, 0.000000e+00

;-------------------------------------------------------------------------------
; (4) Two eligible direct single-child pairs (pairA, pairB) share one ancestor,
; making the breadth-first list non-linear. The current pass emits the analysis
; remark, then bails at the linearity check, leaving both pairs unchanged. The
; follow-on candidate-formation implementation, which transforms at most one
; fallback candidate per nest, will interchange exactly one of them.
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

; Both pairs keep their original order and address expressions.
; IR-LABEL: define void @two_candidate_pairs_one_fallback(
; IR:         %sumA.j = phi double [ %sumA.i, %pairA.outer.header ], [ %sumA.j.next, %pairA.inner ]
; IR:         %idxA = getelementptr inbounds [1335 x double], ptr %A, i64 %jA, i64 %iA
; IR:         %sumA.j.next = fadd reassoc double %sumA.j, %a
; IR:         %sumA.i.lcssa = phi double [ %sumA.j.next, %pairA.inner ]
; IR:         %sumA.live = phi double [ %sumA.i.lcssa, %pairA.outer.latch ]
; IR:         store double %sumA.live, ptr %R, align 8
; IR:         %sumB.j = phi double [ %sumB.i, %pairB.outer.header ], [ %sumB.j.next, %pairB.inner ]
; IR:         %idxB = getelementptr inbounds [1335 x double], ptr %B, i64 %jB, i64 %iB
; IR:         %sumB.j.next = fadd reassoc double %sumB.j, %b
; IR:         %sumB.i.lcssa = phi double [ %sumB.j.next, %pairB.inner ]
; IR:         %sumB.live = phi double [ %sumB.i.lcssa, %pairB.outer.latch ]
; IR:         store double %sumB.live, ptr %rB, align 8

;-------------------------------------------------------------------------------
; (5a) Lasting negative: a partly-exact reduction set. sumA is reassociable but
; sumB is a strict fadd, so even though this is a plain admissible 2-deep nest
; the current pass refuses it (UnsupportedPHIOuter). It must stay refused after
; the follow-on candidate-formation implementation as well -- reassociation is
; required on every reordered recurrence.
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
; embedded beside a sibling loop so the current pass bails at the linearity
; check regardless of profitability -- the point this precommit pins is simply
; that it is not transformed. Once the follow-on candidate-formation
; implementation lands, its fallback reaches this pair and default
; profitability declines the dynamic stride, so it must remain out of scope.
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
; the flat list is non-linear and the current pass bails at the linearity check
; after the analysis remark. The follow-on candidate-formation fallback reaches
; this direct i/j pair but must decline it -- reordering a strict fadd changes
; the result, so reassoc is required on every reordered recurrence. The result
; is stored (real live-out).
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

;-------------------------------------------------------------------------------
; (5d) Lasting negative inside the same fallback-triggering shape: a partly-exact
; reduction pair (sumA reassociable, sumB a strict fadd) nested under ancestor k
; beside a sibling, so the current pass again bails at the linearity check. The
; follow-on candidate-formation fallback reaches the i/j pair but must decline
; it -- every reordered recurrence must be reassociable and sumB is not. Both
; results are stored.
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
; is non-linear. The current pass emits the analysis remark and bails at the
; linearity check. The follow-on candidate-formation fallback initially selects
; only a leaf inner loop, so this non-leaf candidate must remain skipped. The
; candidate store is observable.
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
