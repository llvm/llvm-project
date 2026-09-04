; Behavior test for the surrounding dependence context of an inner-subnest
; candidate in LoopInterchange.
;
; This file is entirely hand-maintained. Do NOT run update_test_checks.py on it.
;
; Every function has a true ancestor loop `k`, an adjacent candidate pair
; `i`(outer)/`j`(inner), and a *sibling* loop `s` with its own memory traffic.
; Because the ancestor has two child loops the breadth-first LoopNest list is
; not a single linear chain, so LoopInterchangePass::run routes it to the
; inner-subnest fallback. The fallback builds the candidate's direction matrix
; over the full k/i/j ancestor chain, collecting memory only from the candidate
; outer loop (so the sibling `s` traffic is excluded), and applies the
; conservative ancestor-prefix rule.
;
; Four access patterns exercise the ancestor-prefix cases: known-forward and
; equal/legal are legal to interchange, while unknown and equal/unsafe are
; rejected. `sibling_store_not_a_candidate_dimension` overlaps sibling traffic
; with the candidate array and covers the excluded-sibling placement.
; Its contrast, `folded_store_in_candidate_rejects`, places the extra store
; inside the candidate and verifies that candidate memory is collected.
;
; Under default profitability the candidate arrays are indexed with `j`
; innermost (unit stride), so interchange is legal-but-unprofitable and the IR
; is unchanged; the IR run pins that. The LEGAL run isolates legality from
; profitability with -loop-interchange-profitabilities=ignore: the two legal
; prefixes and the sibling-overlap case are interchanged, while the
; unknown-ancestor, unsafe-column, and in-candidate contrast cases are rejected.
; The DA run pins that the unknown-ancestor dependence is real (not confused).
;
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -S 2>&1 | FileCheck %s --check-prefix=IR
;
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -pass-remarks=loop-interchange -pass-remarks-missed=loop-interchange \
; RUN:     -pass-remarks-output=%t -disable-output
; RUN: FileCheck %s --check-prefix=YAML --input-file=%t \
; RUN:     --implicit-check-not='Computed dependence info'
;
; Legality isolated from profitability. The two legal prefixes and the
; sibling-overlap case reach Passed selection with candidate matrices that
; exclude the sibling. The unknown ancestor, unsafe candidate, and in-candidate
; contrast cases are rejected. The analysis verifiers must pass after each
; selected transform.
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-profitabilities=ignore \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -pass-remarks=loop-interchange -pass-remarks-missed=loop-interchange \
; RUN:     -pass-remarks-output=%t.ignore -disable-output
; RUN: FileCheck %s --check-prefix=LEGAL --input-file=%t.ignore
;
; The unknown-ancestor fixture must expose a *real*, non-confused surrounding
; dependence whose ancestor (k) column is unknown while the selected i/j columns
; are known and legal in isolation. Prove that independently of LoopInterchange
; with the dependence-analysis printer: DA reports a genuine flow/anti
; dependence whose outermost (k) direction is `*` -- not `confused!`.
; (LoopInterchange's own matrix normalizes this exact CF[j-1][i] = CF[j][i]
; shape to `* = <`; see the all_eq_lt case in legality-check.ll.) The fallback's
; conservative prefix rule rejects this `* = <` because the ancestor direction
; is unknown. Dropping the ancestor would expose only the legal `[= <]` i/j
; columns and interchange the pair, matching the standard path's policy for
; this matrix; this fixture pins the fallback's stricter decision.
; RUN: opt < %s -passes='print<da>' -aa-pipeline=basic-aa -disable-output 2>&1 \
; RUN:     | FileCheck %s --check-prefix=DA

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@KF = global [8 x [8 x [8 x double]]] zeroinitializer
@EQ = global [8 x [8 x [8 x double]]] zeroinitializer
@US = global [8 x [8 x [8 x double]]] zeroinitializer
@CF = global [8 x [8 x double]] zeroinitializer
@SB = global [8 x [8 x double]] zeroinitializer

; Default profitability: each nest reaches the inner-subnest fallback. The two
; legal prefixes and the sibling-overlap case pass legality but are declined by
; profitability (unit-stride `j`); the unknown ancestor prefix and the unsafe
; candidate columns are rejected before profitability.
; YAML:      --- !Missed
; YAML:      Name:            InterchangeNotProfitable
; YAML:      Function:        dep_known_forward_ancestor
; YAML:      --- !Missed
; YAML:      Name:            InterchangeNotProfitable
; YAML:      Function:        dep_equal_legal_ancestor
; The unknown ancestor prefix is rejected by the conservative prefix rule.
; YAML:      --- !Missed
; YAML:      Name:            FallbackUnknownContext
; YAML:      Function:        dep_unknown_ancestor
; The unsafe candidate columns are rejected by the existing pair legality.
; YAML:      --- !Missed
; YAML:      Name:            Dependence
; YAML:      Function:        dep_equal_unsafe_ancestor
; YAML:      --- !Missed
; YAML:      Name:            InterchangeNotProfitable
; YAML:      Function:        sibling_store_not_a_candidate_dimension
; The contrast fixture: the extra store is inside the candidate loop, so it is
; collected and rejects the swap on dependence grounds (not profitability).
; YAML:      --- !Missed
; YAML:      Name:            Dependence
; YAML:      Function:        folded_store_in_candidate_rejects

; With profitability ignored, legality alone decides. The known-forward and
; equal/legal prefixes interchange; the sibling-overlap case interchanges too,
; covering overlapping traffic outside the candidate matrix. The unknown
; ancestor prefix and the unsafe candidate columns are still rejected.
; LEGAL:      --- !Passed
; LEGAL:      Name:            Interchanged
; LEGAL:      Function:        dep_known_forward_ancestor
; LEGAL:      --- !Passed
; LEGAL:      Name:            Interchanged
; LEGAL:      Function:        dep_equal_legal_ancestor
; LEGAL:      --- !Missed
; LEGAL:      Name:            FallbackUnknownContext
; LEGAL:      Function:        dep_unknown_ancestor
; LEGAL:      --- !Missed
; LEGAL:      Name:            Dependence
; LEGAL:      Function:        dep_equal_unsafe_ancestor
; LEGAL:      --- !Passed
; LEGAL:      Name:            Interchanged
; LEGAL:      Function:        sibling_store_not_a_candidate_dimension
; The contrast fixture rejects even with profitability ignored: legality alone
; declines it because the in-candidate j-invariant store carries `*` in j.
; LEGAL:      --- !Missed
; LEGAL:      Name:            Dependence
; LEGAL:      Function:        folded_store_in_candidate_rejects
; LEGAL-NOT:  Function:        folded_store_in_candidate_rejects

;-------------------------------------------------------------------------------
; Known-forward ancestor prefix: the store to KF[k+1][i][j] is read at KF[k][i][j]
; on the next k iteration, a lexicographically forward carry on the ancestor.
; That forward prefix is decisive, so the i/j swap is legal (LEGAL run); under
; default profitability it is unit-stride and therefore declined.
;-------------------------------------------------------------------------------
define void @dep_known_forward_ancestor() {
entry:
  br label %k.header

k.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %k.latch ]
  br label %i.header

i.header:
  %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
  br label %j.body

j.body:
  %j = phi i64 [ 0, %i.header ], [ %j.next, %j.body ]
  %kp1 = add i64 %k, 1
  %ld.idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @KF, i64 0, i64 %k, i64 %i, i64 %j
  %v = load double, ptr %ld.idx, align 8
  %nv = fadd double %v, 1.000000e+00
  %st.idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @KF, i64 0, i64 %kp1, i64 %i, i64 %j
  store double %nv, ptr %st.idx, align 8
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 7
  br i1 %j.ec, label %i.latch, label %j.body

i.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 7
  br i1 %i.ec, label %sib.preheader, label %i.header

sib.preheader:
  br label %sib.header

sib.header:
  %s = phi i64 [ 0, %sib.preheader ], [ %s.next, %sib.header ]
  %s.idx = getelementptr inbounds [8 x [8 x double]], ptr @SB, i64 0, i64 %k, i64 %s
  %s.v = load double, ptr %s.idx, align 8
  %s.nv = fadd double %s.v, 1.000000e+00
  store double %s.nv, ptr %s.idx, align 8
  %s.next = add i64 %s, 1
  %s.ec = icmp eq i64 %s.next, 7
  br i1 %s.ec, label %k.latch, label %sib.header

k.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 7
  br i1 %k.ec, label %exit, label %k.header

exit:
  ret void
}

; Original k/i/j nesting and the KF[k]/KF[k+1] carry are preserved, and the
; sibling still writes SB[k][s].
; IR-LABEL: define void @dep_known_forward_ancestor(
; IR:         %k = phi i64 [ 0, %entry ], [ %k.next, %k.latch ]
; IR:         %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
; IR:         %j = phi i64 [ 0, %i.header ], [ %j.next, %j.body ]
; IR:         %ld.idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @KF, i64 0, i64 %k, i64 %i, i64 %j
; IR:         %st.idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @KF, i64 0, i64 %kp1, i64 %i, i64 %j
; IR:         store double %nv, ptr %st.idx, align 8
; IR:         %s = phi i64 [ 0, %sib.preheader ], [ %s.next, %sib.header ]
; IR:         %s.idx = getelementptr inbounds [8 x [8 x double]], ptr @SB, i64 0, i64 %k, i64 %s

;-------------------------------------------------------------------------------
; Equal/legal ancestor prefix: a read-modify-write of the same EQ[k][i][j], a
; loop-independent (equal) dependence. The equal ancestor prefix delegates to
; the i/j columns, which are legal to swap (LEGAL run).
;-------------------------------------------------------------------------------
define void @dep_equal_legal_ancestor() {
entry:
  br label %k.header

k.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %k.latch ]
  br label %i.header

i.header:
  %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
  br label %j.body

j.body:
  %j = phi i64 [ 0, %i.header ], [ %j.next, %j.body ]
  %idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @EQ, i64 0, i64 %k, i64 %i, i64 %j
  %v = load double, ptr %idx, align 8
  %nv = fadd double %v, 1.000000e+00
  store double %nv, ptr %idx, align 8
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 7
  br i1 %j.ec, label %i.latch, label %j.body

i.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 7
  br i1 %i.ec, label %sib.preheader, label %i.header

sib.preheader:
  br label %sib.header

sib.header:
  %s = phi i64 [ 0, %sib.preheader ], [ %s.next, %sib.header ]
  %s.idx = getelementptr inbounds [8 x [8 x double]], ptr @SB, i64 0, i64 %k, i64 %s
  %s.v = load double, ptr %s.idx, align 8
  %s.nv = fadd double %s.v, 1.000000e+00
  store double %s.nv, ptr %s.idx, align 8
  %s.next = add i64 %s, 1
  %s.ec = icmp eq i64 %s.next, 7
  br i1 %s.ec, label %k.latch, label %sib.header

k.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 7
  br i1 %k.ec, label %exit, label %k.header

exit:
  ret void
}

; IR-LABEL: define void @dep_equal_legal_ancestor(
; IR:         %k = phi i64 [ 0, %entry ], [ %k.next, %k.latch ]
; IR:         %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
; IR:         %j = phi i64 [ 0, %i.header ], [ %j.next, %j.body ]
; IR:         %idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @EQ, i64 0, i64 %k, i64 %i, i64 %j
; IR:         store double %nv, ptr %idx, align 8
; IR:         %s.idx = getelementptr inbounds [8 x [8 x double]], ptr @SB, i64 0, i64 %k, i64 %s

;-------------------------------------------------------------------------------
; Unknown ancestor prefix: the surrounding k loop does not index CF at all, so
; the ancestor column of the candidate pair's dependence is unknown (`*`), while
; the selected i/j columns are perfectly known -- equal in i and unit-carried in
; j (CF[j-1][i] = CF[j][i], the all_eq_lt shape from legality-check.ll whose
; interchange matrix is `* = <`). Dependence analysis returns a real dependence,
; not `confused!` (see the DA run). The fallback's conservative prefix rule
; rejects this pair (FallbackUnknownContext) because the ancestor direction is
; unknown. Dropping the ancestor would expose only the legal `[= <]` i/j
; columns and interchange the pair, as the standard path does for this matrix;
; this fixture pins the fallback's stricter decision.
;-------------------------------------------------------------------------------
define void @dep_unknown_ancestor() {
entry:
  br label %k.header

k.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %k.latch ]
  br label %i.header

i.header:
  %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
  br label %j.body

j.body:
  %j = phi i64 [ 1, %i.header ], [ %j.next, %j.body ]
  %jm1 = sub i64 %j, 1
  %ld.idx = getelementptr inbounds [8 x [8 x double]], ptr @CF, i64 0, i64 %j, i64 %i
  %v = load double, ptr %ld.idx, align 8
  %nv = fadd double %v, 1.000000e+00
  %st.idx = getelementptr inbounds [8 x [8 x double]], ptr @CF, i64 0, i64 %jm1, i64 %i
  store double %nv, ptr %st.idx, align 8
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 8
  br i1 %j.ec, label %i.latch, label %j.body

i.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 7
  br i1 %i.ec, label %sib.preheader, label %i.header

sib.preheader:
  br label %sib.header

sib.header:
  %s = phi i64 [ 0, %sib.preheader ], [ %s.next, %sib.header ]
  %s.idx = getelementptr inbounds [8 x [8 x double]], ptr @SB, i64 0, i64 %k, i64 %s
  %s.v = load double, ptr %s.idx, align 8
  %s.nv = fadd double %s.v, 1.000000e+00
  store double %s.nv, ptr %s.idx, align 8
  %s.next = add i64 %s, 1
  %s.ec = icmp eq i64 %s.next, 7
  br i1 %s.ec, label %k.latch, label %sib.header

k.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 7
  br i1 %k.ec, label %exit, label %k.header

exit:
  ret void
}

; The ancestor k does not index CF, so the surrounding context is unknown; the
; candidate reads CF[j][i] and writes CF[j-1][i] in original i/j order, and the
; sibling still writes SB[k][s]. Nothing is interchanged.
; IR-LABEL: define void @dep_unknown_ancestor(
; IR:         %k = phi i64 [ 0, %entry ], [ %k.next, %k.latch ]
; IR:         %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
; IR:         %j = phi i64 [ 1, %i.header ], [ %j.next, %j.body ]
; IR:         %ld.idx = getelementptr inbounds [8 x [8 x double]], ptr @CF, i64 0, i64 %j, i64 %i
; IR:         %st.idx = getelementptr inbounds [8 x [8 x double]], ptr @CF, i64 0, i64 %jm1, i64 %i
; IR:         store double %nv, ptr %st.idx, align 8
; IR:         %s.idx = getelementptr inbounds [8 x [8 x double]], ptr @SB, i64 0, i64 %k, i64 %s

; DA reports a genuine (non-confused) anti-dependence for the CF read/write pair.
; The ancestor is scalar (`S`, mapped conservatively to `*` by
; LoopInterchange), while the selected i/j levels have known distances 0/1.
; The normalized interchange matrix is `* = <`.
; DA-LABEL: 'dep_unknown_ancestor'
; DA:       da analyze - anti [S 0 1]!

;-------------------------------------------------------------------------------
; Equal ancestor prefix, unsafe selected columns: US[k][i][j+1] is written from
; US[k][i+1][j], a cross i/j carry that is not safe to interchange even though
; the ancestor prefix is equal. The i/j columns themselves are rejected by the
; existing pair legality (LEGAL run).
;-------------------------------------------------------------------------------
define void @dep_equal_unsafe_ancestor() {
entry:
  br label %k.header

k.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %k.latch ]
  br label %i.header

i.header:
  %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
  %ip1 = add i64 %i, 1
  br label %j.body

j.body:
  %j = phi i64 [ 0, %i.header ], [ %j.next, %j.body ]
  %jp1 = add i64 %j, 1
  %ld.idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @US, i64 0, i64 %k, i64 %ip1, i64 %j
  %v = load double, ptr %ld.idx, align 8
  %nv = fadd double %v, 1.000000e+00
  %st.idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @US, i64 0, i64 %k, i64 %i, i64 %jp1
  store double %nv, ptr %st.idx, align 8
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 7
  br i1 %j.ec, label %i.latch, label %j.body

i.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 7
  br i1 %i.ec, label %sib.preheader, label %i.header

sib.preheader:
  br label %sib.header

sib.header:
  %s = phi i64 [ 0, %sib.preheader ], [ %s.next, %sib.header ]
  %s.idx = getelementptr inbounds [8 x [8 x double]], ptr @SB, i64 0, i64 %k, i64 %s
  %s.v = load double, ptr %s.idx, align 8
  %s.nv = fadd double %s.v, 1.000000e+00
  store double %s.nv, ptr %s.idx, align 8
  %s.next = add i64 %s, 1
  %s.ec = icmp eq i64 %s.next, 7
  br i1 %s.ec, label %k.latch, label %sib.header

k.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 7
  br i1 %k.ec, label %exit, label %k.header

exit:
  ret void
}

; IR-LABEL: define void @dep_equal_unsafe_ancestor(
; IR:         %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
; IR:         %j = phi i64 [ 0, %i.header ], [ %j.next, %j.body ]
; IR:         %ld.idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @US, i64 0, i64 %k, i64 %ip1, i64 %j
; IR:         %st.idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @US, i64 0, i64 %k, i64 %i, i64 %jp1
; IR:         store double %nv, ptr %st.idx, align 8

;-------------------------------------------------------------------------------
; The sibling loop stores into the *same* array KF that the candidate reads, but
; on a different (diagonal) index. The fallback collects the candidate matrix
; only from the candidate outer loop, so this sibling store is excluded and the
; candidate interchanges with profitability ignored (LEGAL run); under default
; profitability it is unit-stride and unchanged (IR run).
;
; The contrast fixture folded_store_in_candidate_rejects places an extra store
; inside the candidate j loop, where it is collected. That store's j-invariant
; self-output dependence carries `*` in the j column and defeats interchange.
; The two fixtures cover sibling and in-candidate placements with different
; access patterns; they do not assert an if-and-only-if relationship.
;-------------------------------------------------------------------------------
define void @sibling_store_not_a_candidate_dimension() {
entry:
  br label %k.header

k.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %k.latch ]
  br label %i.header

i.header:
  %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
  br label %j.body

j.body:
  %j = phi i64 [ 0, %i.header ], [ %j.next, %j.body ]
  %idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @KF, i64 0, i64 %k, i64 %i, i64 %j
  %v = load double, ptr %idx, align 8
  %nv = fadd double %v, 1.000000e+00
  store double %nv, ptr %idx, align 8
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 7
  br i1 %j.ec, label %i.latch, label %j.body

i.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 7
  br i1 %i.ec, label %sib.preheader, label %i.header

sib.preheader:
  br label %sib.header

sib.header:
  %s = phi i64 [ 0, %sib.preheader ], [ %s.next, %sib.header ]
  %sib.idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @KF, i64 0, i64 %k, i64 %s, i64 %s
  store double 1.000000e+00, ptr %sib.idx, align 8
  %s.next = add i64 %s, 1
  %s.ec = icmp eq i64 %s.next, 7
  br i1 %s.ec, label %k.latch, label %sib.header

k.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 7
  br i1 %k.ec, label %exit, label %k.header

exit:
  ret void
}

; The candidate KF[k][i][j] access and the sibling diagonal store KF[k][s][s]
; both remain, with the pair in original order.
; IR-LABEL: define void @sibling_store_not_a_candidate_dimension(
; IR:         %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
; IR:         %j = phi i64 [ 0, %i.header ], [ %j.next, %j.body ]
; IR:         %idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @KF, i64 0, i64 %k, i64 %i, i64 %j
; IR:         store double %nv, ptr %idx, align 8
; IR:         %sib.idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @KF, i64 0, i64 %k, i64 %s, i64 %s
; IR:         store double 1.000000e+00, ptr %sib.idx, align 8

;-------------------------------------------------------------------------------
; Contrast fixture for the sibling-exclusion claim above. The candidate does the
; same KF[k][i][j] read-modify-write, but an extra store to the j-invariant
; address KF[k][i][0] now sits *inside* the candidate j loop (not in a sibling).
; Because it is inside the candidate outer loop, it is collected into the matrix,
; and its output self-dependence carries `*` in the j column (the same location
; is written on every j), so the i/j swap is rejected on dependence grounds --
; even with profitability ignored. This covers the in-candidate placement.
; Together with the sibling fixture it covers both placements, but the two
; fixtures use different access patterns and do not establish an if-and-only-if
; relationship.
;-------------------------------------------------------------------------------
define void @folded_store_in_candidate_rejects() {
entry:
  br label %k.header

k.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %k.latch ]
  br label %i.header

i.header:
  %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
  br label %j.body

j.body:
  %j = phi i64 [ 0, %i.header ], [ %j.next, %j.body ]
  %idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @KF, i64 0, i64 %k, i64 %i, i64 %j
  %v = load double, ptr %idx, align 8
  %nv = fadd double %v, 1.000000e+00
  store double %nv, ptr %idx, align 8
  %inv.idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @KF, i64 0, i64 %k, i64 %i, i64 0
  store double %nv, ptr %inv.idx, align 8
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 7
  br i1 %j.ec, label %i.latch, label %j.body

i.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 7
  br i1 %i.ec, label %sib.preheader, label %i.header

sib.preheader:
  br label %sib.header

sib.header:
  %s = phi i64 [ 0, %sib.preheader ], [ %s.next, %sib.header ]
  %s.idx = getelementptr inbounds [8 x [8 x double]], ptr @SB, i64 0, i64 %k, i64 %s
  %s.v = load double, ptr %s.idx, align 8
  %s.nv = fadd double %s.v, 1.000000e+00
  store double %s.nv, ptr %s.idx, align 8
  %s.next = add i64 %s, 1
  %s.ec = icmp eq i64 %s.next, 7
  br i1 %s.ec, label %k.latch, label %sib.header

k.latch:
  %k.next = add i64 %k, 1
  %k.ec = icmp eq i64 %k.next, 7
  br i1 %k.ec, label %exit, label %k.header

exit:
  ret void
}

; The candidate keeps its original i/j order: the extra j-invariant store
; KF[k][i][0] inside the j loop is collected and its `*`-in-j self output
; dependence declines the swap (contrast with the excluded sibling above).
; IR-LABEL: define void @folded_store_in_candidate_rejects(
; IR:         %i = phi i64 [ 0, %k.header ], [ %i.next, %i.latch ]
; IR:         %j = phi i64 [ 0, %i.header ], [ %j.next, %j.body ]
; IR:         %idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @KF, i64 0, i64 %k, i64 %i, i64 %j
; IR:         %inv.idx = getelementptr inbounds [8 x [8 x [8 x double]]], ptr @KF, i64 0, i64 %k, i64 %i, i64 0
