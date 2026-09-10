; Cache profitability for a fallback pair must use that pair's root-to-leaf
; ancestor chain and exclude references from disjoint sibling subtrees.
;
; RUN: llvm-extract -S -func=disjoint_costmodel %s -o %t.disjoint
; RUN: opt < %t.disjoint -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-profitabilities=cache \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -pass-remarks-output=%t.disjoint.yaml -disable-output
; RUN: FileCheck %s --check-prefix=DISJOINT \
; RUN:     --input-file=%t.disjoint.yaml --implicit-check-not=Interchanged
;
; RUN: llvm-extract -S -func=control_same_pair_alone %s -o %t.control-negative
; RUN: opt < %t.control-negative -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-profitabilities=cache \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -pass-remarks-output=%t.control-negative.yaml -disable-output
; RUN: FileCheck %s --check-prefix=CONTROL-NEGATIVE \
; RUN:     --input-file=%t.control-negative.yaml \
; RUN:     --implicit-check-not=Interchanged
;
; RUN: llvm-extract -S -func=fallback_cache_profitable %s -o %t.fallback-positive
; RUN: opt < %t.fallback-positive -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-profitabilities=cache \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -pass-remarks-output=%t.fallback-positive.yaml -disable-output
; RUN: FileCheck %s --check-prefix=FALLBACK-POSITIVE \
; RUN:     --input-file=%t.fallback-positive.yaml
;
; RUN: llvm-extract -S -func=linear_cache_profitable %s -o %t.linear-positive
; RUN: opt < %t.linear-positive -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-profitabilities=cache \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -pass-remarks-output=%t.linear-positive.yaml -disable-output
; RUN: FileCheck %s --check-prefix=LINEAR-POSITIVE \
; RUN:     --input-file=%t.linear-positive.yaml
;
; RUN: llvm-extract -S -func=two_equal_depth_leaves %s -o %t.equal-depth
; RUN: opt < %t.equal-depth -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-profitabilities=cache \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -pass-remarks-output=%t.equal-depth.yaml -disable-output
; RUN: FileCheck %s --check-prefix=EQUAL-DEPTH \
; RUN:     --input-file=%t.equal-depth.yaml
;
; RUN: opt < %s -passes='loop(loop-interchange),print<loops>' \
; RUN:     -cache-line-size=64 -loop-interchange-profitabilities=cache \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=LOOPS

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@X = global [4 x [1335 x [100 x double]]] zeroinitializer
@Y = global [4 x [4 x [1335 x [1335 x double]]]] zeroinitializer
@P = global [4 x [128 x [128 x double]]] zeroinitializer
@S = global [4 x [128 x double]] zeroinitializer

; The d/e pair is rejected for dependence. The fallback then evaluates a/b,
; whose inner j loop is already unit stride. The disjoint c/d/e subtree must
; not make a/b appear profitable.
;
; DISJOINT:      --- !Missed
; DISJOINT:      Name:            Dependence
; DISJOINT-NEXT: Function:        disjoint_costmodel
; DISJOINT:      --- !Missed
; DISJOINT:      Name:            InterchangeNotProfitable
; DISJOINT-NEXT: Function:        disjoint_costmodel
; DISJOINT:        - String:          Interchanging loops is not considered to improve cache locality nor vectorization.
define void @disjoint_costmodel() {
entry:
  br label %r.header

r.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %r.latch ]
  br label %a.header

a.header:
  %i = phi i64 [ 0, %r.header ], [ %i.next, %a.latch ]
  br label %b.body

b.body:
  %j = phi i64 [ 0, %a.header ], [ %j.next, %b.body ]
  %xp = getelementptr inbounds [4 x [1335 x [100 x double]]],
      ptr @X, i64 0, i64 %k, i64 %i, i64 %j
  %xv = load double, ptr %xp, align 8
  %xa = fadd double %xv, 1.000000e+00
  store double %xa, ptr %xp, align 8
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 100
  br i1 %j.done, label %a.latch, label %b.body

a.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 1335
  br i1 %i.done, label %c.header, label %a.header

c.header:
  %p = phi i64 [ 0, %a.latch ], [ %p.next, %c.latch ]
  br label %d.header

d.header:
  %q = phi i64 [ 1, %c.header ], [ %q.next, %d.latch ]
  br label %e.body

e.body:
  %w = phi i64 [ 0, %d.header ], [ %w.next, %e.body ]
  %q.prev = add i64 %q, -1
  %w.next.index = add i64 %w, 1
  %ysrc = getelementptr inbounds [4 x [4 x [1335 x [1335 x double]]]],
      ptr @Y, i64 0, i64 %k, i64 %p, i64 %q.prev, i64 %w.next.index
  %yv = load double, ptr %ysrc, align 8
  %ya = fadd double %yv, 1.000000e+00
  %ydst = getelementptr inbounds [4 x [4 x [1335 x [1335 x double]]]],
      ptr @Y, i64 0, i64 %k, i64 %p, i64 %q, i64 %w
  store double %ya, ptr %ydst, align 8
  %w.next = add i64 %w, 1
  %w.done = icmp eq i64 %w.next, 1334
  br i1 %w.done, label %d.latch, label %e.body

d.latch:
  %q.next = add i64 %q, 1
  %q.done = icmp eq i64 %q.next, 1335
  br i1 %q.done, label %c.latch, label %d.header

c.latch:
  %p.next = add i64 %p, 1
  %p.done = icmp eq i64 %p.next, 4
  br i1 %p.done, label %r.latch, label %c.header

r.latch:
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 4
  br i1 %k.done, label %exit, label %r.header

exit:
  ret void
}

; The identical pair on the standard path has a complete cache analysis and
; declines because interchange would not improve locality.
;
; CONTROL-NEGATIVE:      --- !Missed
; CONTROL-NEGATIVE:      Name:            InterchangeNotProfitable
; CONTROL-NEGATIVE-NEXT: Function:        control_same_pair_alone
; CONTROL-NEGATIVE:        - String:          Interchanging loops is not considered to improve cache locality nor vectorization.
define void @control_same_pair_alone() {
entry:
  br label %r.header

r.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %r.latch ]
  br label %a.header

a.header:
  %i = phi i64 [ 0, %r.header ], [ %i.next, %a.latch ]
  br label %b.body

b.body:
  %j = phi i64 [ 0, %a.header ], [ %j.next, %b.body ]
  %xp = getelementptr inbounds [4 x [1335 x [100 x double]]],
      ptr @X, i64 0, i64 %k, i64 %i, i64 %j
  %xv = load double, ptr %xp, align 8
  %xa = fadd double %xv, 1.000000e+00
  store double %xa, ptr %xp, align 8
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 100
  br i1 %j.done, label %a.latch, label %b.body

a.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 1335
  br i1 %i.done, label %r.latch, label %a.header

r.latch:
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 4
  br i1 %k.done, label %exit, label %r.header

exit:
  ret void
}

; The candidate leaf supplies the candidate chain's cache reference groups, so
; the fallback keeps the same positive cache verdict as the linear control.
;
; FALLBACK-POSITIVE:      --- !Passed
; FALLBACK-POSITIVE:      Name:            Interchanged
; FALLBACK-POSITIVE-NEXT: Function:        fallback_cache_profitable
define void @fallback_cache_profitable() {
entry:
  br label %root.header

root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %root.header ], [ %i.next, %outer.latch ]
  br label %inner.body

inner.body:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.body ]
  %p = getelementptr inbounds [4 x [128 x [128 x double]]],
      ptr @P, i64 0, i64 %k, i64 %j, i64 %i
  %v = load double, ptr %p, align 8
  %next = fadd double %v, 1.000000e+00
  store double %next, ptr %p, align 8
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 128
  br i1 %j.done, label %outer.latch, label %inner.body

outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 128
  br i1 %i.done, label %sibling.preheader, label %outer.header

sibling.preheader:
  br label %sibling.body

sibling.body:
  %s = phi i64 [ 0, %sibling.preheader ], [ %s.next, %sibling.body ]
  %sp = getelementptr inbounds [4 x [128 x double]],
      ptr @S, i64 0, i64 %k, i64 %s
  store double 1.000000e+00, ptr %sp, align 8
  %s.next = add i64 %s, 1
  %s.done = icmp eq i64 %s.next, 128
  br i1 %s.done, label %root.latch, label %sibling.body

root.latch:
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 4
  br i1 %k.done, label %exit, label %root.header

exit:
  ret void
}

; LINEAR-POSITIVE:      --- !Passed
; LINEAR-POSITIVE:      Name:            Interchanged
; LINEAR-POSITIVE-NEXT: Function:        linear_cache_profitable
define void @linear_cache_profitable() {
entry:
  br label %root.header

root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %root.header ], [ %i.next, %outer.latch ]
  br label %inner.body

inner.body:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.body ]
  %p = getelementptr inbounds [4 x [128 x [128 x double]]],
      ptr @P, i64 0, i64 %k, i64 %j, i64 %i
  %v = load double, ptr %p, align 8
  %next = fadd double %v, 1.000000e+00
  store double %next, ptr %p, align 8
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 128
  br i1 %j.done, label %outer.latch, label %inner.body

outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 128
  br i1 %i.done, label %root.latch, label %outer.header

root.latch:
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 4
  br i1 %k.done, label %exit, label %root.header

exit:
  ret void
}

; Both candidate leaves have the same depth. The first pair is not profitable;
; the second pair is profitable from its own reference groups and interchanges.
;
; EQUAL-DEPTH:      --- !Missed
; EQUAL-DEPTH:      Name:            InterchangeNotProfitable
; EQUAL-DEPTH-NEXT: Function:        two_equal_depth_leaves
; EQUAL-DEPTH:      --- !Passed
; EQUAL-DEPTH:      Name:            Interchanged
; EQUAL-DEPTH-NEXT: Function:        two_equal_depth_leaves
define void @two_equal_depth_leaves() {
entry:
  br label %root.header

root.header:
  %k = phi i64 [ 0, %entry ], [ %k.next, %root.latch ]
  br label %first.outer

first.outer:
  %i = phi i64 [ 0, %root.header ], [ %i.next, %first.outer.latch ]
  br label %first.inner

first.inner:
  %j = phi i64 [ 0, %first.outer ], [ %j.next, %first.inner ]
  %xp = getelementptr inbounds [4 x [1335 x [100 x double]]],
      ptr @X, i64 0, i64 %k, i64 %i, i64 %j
  %xv = load double, ptr %xp, align 8
  %xa = fadd double %xv, 1.000000e+00
  store double %xa, ptr %xp, align 8
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 100
  br i1 %j.done, label %first.outer.latch, label %first.inner

first.outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 1335
  br i1 %i.done, label %second.preheader, label %first.outer

second.preheader:
  br label %second.outer

second.outer:
  %i2 = phi i64 [ 0, %second.preheader ], [ %i2.next, %second.outer.latch ]
  br label %second.inner

second.inner:
  %j2 = phi i64 [ 0, %second.outer ], [ %j2.next, %second.inner ]
  %pp = getelementptr inbounds [4 x [128 x [128 x double]]],
      ptr @P, i64 0, i64 %k, i64 %j2, i64 %i2
  %pv = load double, ptr %pp, align 8
  %pa = fadd double %pv, 1.000000e+00
  store double %pa, ptr %pp, align 8
  %j2.next = add i64 %j2, 1
  %j2.done = icmp eq i64 %j2.next, 128
  br i1 %j2.done, label %second.outer.latch, label %second.inner

second.outer.latch:
  %i2.next = add i64 %i2, 1
  %i2.done = icmp eq i64 %i2.next, 128
  br i1 %i2.done, label %root.latch, label %second.outer

root.latch:
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 4
  br i1 %k.done, label %exit, label %root.header

exit:
  ret void
}

; Cache-negative fallback candidates keep their original order; profitable
; fallback candidates interchange.
; LOOPS-LABEL: Loop info for function 'disjoint_costmodel':
; LOOPS:         Loop at depth 1 containing: %r.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %a.header<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %b.body<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %c.header<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %d.header<header>
; LOOPS-NEXT:          Loop at depth 4 containing: %e.body<header>
; LOOPS-LABEL: Loop info for function 'control_same_pair_alone':
; LOOPS:         Loop at depth 1 containing: %r.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %a.header<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %b.body<header>
; LOOPS-LABEL: Loop info for function 'fallback_cache_profitable':
; LOOPS:         Loop at depth 1 containing: %root.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %inner.body<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %outer.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %sibling.body<header>
; LOOPS-LABEL: Loop info for function 'linear_cache_profitable':
; LOOPS:         Loop at depth 1 containing: %root.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %inner.body<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %outer.header<header>
; LOOPS-LABEL: Loop info for function 'two_equal_depth_leaves':
; LOOPS:         Loop at depth 1 containing: %root.header<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %first.outer<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %first.inner<header>
; LOOPS-NEXT:      Loop at depth 2 containing: %second.inner<header>
; LOOPS-NEXT:        Loop at depth 3 containing: %second.outer<header>
