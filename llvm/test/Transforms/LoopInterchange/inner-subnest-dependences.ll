; Precommit test for the surrounding dependence context of an inner subnest
; candidate in LoopInterchange.
;
; This file is entirely hand-maintained. Do NOT run update_test_checks.py on it.
;
; Every function has a true ancestor loop `k`, an adjacent candidate pair
; `i`(outer)/`j`(inner), and a *sibling* loop `s` with its own memory traffic.
; Because the ancestor has two child loops the breadth-first LoopNest list is
; not a single linear chain. LoopInterchangePass::run still emits the analysis
; remark (the depth and computability checks pass), and then
; LoopInterchange::run(LoopNest&) bails at its linearity check -- nothing is
; interchanged today. That is all this precommit pins: the analysis remark plus
; the actual unswapped IR (original k/i/j nesting, address expressions, and the
; sibling's separate memory operations left in place).
;
; The four access patterns set up the ancestor-prefix cases that the follow-on
; candidate-formation implementation will distinguish (known-forward,
; equal/legal, unknown, equal/unsafe), and the last function makes the sibling
; store overlap the candidate array to pin that the follow-on implementation
; must NOT fold sibling memory into the candidate's direction matrix. This
; precommit deliberately does not assert those legality outcomes or direction
; vectors; the unmodified pass never computes the candidate pair's matrix here.
;
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -S 2>&1 | FileCheck %s --check-prefix=IR
;
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -pass-remarks=loop-interchange -pass-remarks-missed=loop-interchange \
; RUN:     -pass-remarks-output=%t -disable-output
; RUN: FileCheck %s --check-prefix=YAML --input-file=%t
;
; The unknown-ancestor fixture must expose a *real*, non-confused surrounding
; dependence whose ancestor (k) column is unknown while the selected i/j columns
; are known and legal in isolation. Prove that on the unmodified pass with the
; dependence-analysis printer: DA reports a genuine flow/anti dependence whose
; outermost (k) direction is `*` -- not `confused!`. (LoopInterchange's own
; matrix normalizes this exact CF[j-1][i] = CF[j][i] shape to `* = <`; see the
; all_eq_lt case in legality-check.ll.)
; RUN: opt < %s -passes='print<da>' -aa-pipeline=basic-aa -disable-output 2>&1 \
; RUN:     | FileCheck %s --check-prefix=DA

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@KF = global [8 x [8 x [8 x double]]] zeroinitializer
@EQ = global [8 x [8 x [8 x double]]] zeroinitializer
@US = global [8 x [8 x [8 x double]]] zeroinitializer
@CF = global [8 x [8 x double]] zeroinitializer
@SB = global [8 x [8 x double]] zeroinitializer

; Each nest reaches the transform (analysis remark) but is then rejected as
; non-linear; none is interchanged.
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        dep_known_forward_ancestor
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        dep_equal_legal_ancestor
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        dep_unknown_ancestor
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        dep_equal_unsafe_ancestor
; YAML:      --- !Analysis
; YAML:      Name:            Dependence
; YAML:      Function:        sibling_store_not_a_candidate_dimension

;-------------------------------------------------------------------------------
; Known-forward ancestor prefix: the store to KF[k+1][i][j] is read at KF[k][i][j]
; on the next k iteration, a lexicographically forward carry on the ancestor.
; That forward prefix is decisive for the follow-on candidate-formation
; implementation, which will find the i/j swap legal; here the nest is simply
; not processed (non-linear).
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
; loop-independent (equal) dependence. For the follow-on candidate-formation
; implementation the equal ancestor prefix delegates to the i/j columns, which
; are legal to swap.
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
; not `confused!` (see the DA run). The strict fallback in the follow-on
; candidate-formation implementation must reject this pair because the true
; ancestor context is unknown; an unsound implementation that dropped or
; projected the ancestor away would see only the legal `[= <]` i/j columns and
; wrongly accept. This precommit only pins that the dependence is real and that
; nothing is interchanged.
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
; the ancestor prefix is equal. The i/j columns themselves must reject in the
; follow-on candidate-formation implementation.
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
; on a different (diagonal) index. If the follow-on candidate-formation
; implementation ever collected the candidate pair's direction matrix from the
; whole ancestor subtree it would pull in this sibling store and mis-model it as
; a candidate dimension. Today the nest is simply non-linear and untouched; this
; precommit pins that both the candidate access and the sibling store are
; present and unchanged.
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
