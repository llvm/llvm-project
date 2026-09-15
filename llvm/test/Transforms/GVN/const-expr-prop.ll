; RUN: opt -S -passes=gvn -gvn-const-expr-prop=true < %s | FileCheck %s

declare void @llvm.fake.use(...)

; %a is built solely from %x but is defined *outside* the region dominated by
; the true edge of %c, so propagateEquality's direct-use replacement cannot
; reach it. propagateConstExpressions should clone %a into if.true with %x
; replaced by the known constant 5, and constant-fold the clone.
define i32 @add_outside_dominated_region(i32 %x) {
; CHECK-LABEL: @add_outside_dominated_region(
; CHECK: if.true:
; CHECK-NEXT: ret i32 6
  %a = add i32 %x, 1
  %c = icmp eq i32 %x, 5
  br i1 %c, label %if.true, label %if.false
if.true:
  ret i32 %a
if.false:
  ret i32 0
}

; Same as above, but with the constant on the left-hand side of the compare.
define i32 @add_outside_dominated_region_constant_lhs(i32 %x) {
; CHECK-LABEL: @add_outside_dominated_region_constant_lhs(
; CHECK: if.true:
; CHECK-NEXT: ret i32 6
  %a = add i32 %x, 1
  %c = icmp eq i32 5, %x
  br i1 %c, label %if.true, label %if.false
if.true:
  ret i32 %a
if.false:
  ret i32 0
}

; Cast expressions should also be cloned and folded.
define i64 @zext_outside_dominated_region(i32 %x) {
; CHECK-LABEL: @zext_outside_dominated_region(
; CHECK: if.true:
; CHECK-NEXT: ret i64 5
  %z = zext i32 %x to i64
  %c = icmp eq i32 %x, 5
  br i1 %c, label %if.true, label %if.false
if.true:
  ret i64 %z
if.false:
  ret i64 0
}

; Chains of expressions built solely from %x should be cloned recursively.
define i32 @chained_expr_outside_dominated_region(i32 %x) {
; CHECK-LABEL: @chained_expr_outside_dominated_region(
; CHECK: if.true:
; CHECK-NEXT: ret i32 12
  %a = add i32 %x, 1
  %m = mul i32 %a, 2
  %c = icmp eq i32 %x, 5
  br i1 %c, label %if.true, label %if.false
if.true:
  ret i32 %m
if.false:
  ret i32 0
}

; Switch case edges are also handled, not just conditional branches.
define i32 @switch_case_outside_dominated_region(i32 %x) {
; CHECK-LABEL: @switch_case_outside_dominated_region(
; CHECK: case3:
; CHECK-NEXT: ret i32 13
  %a = add i32 %x, 10
  switch i32 %x, label %default [
  i32 3, label %case3
  ]
case3:
  ret i32 %a
default:
  ret i32 0
}

; The dominated region is intentionally widened beyond Root.getEnd(): an
; expression only reachable through a whole diamond nested inside if.true
; should still be folded.
define i32 @nested_diamond_inside_dominated_region(i32 %x, i1 %arg) {
; CHECK-LABEL: @nested_diamond_inside_dominated_region(
; CHECK: inner.merge:
; CHECK-NEXT: ret i32 6
  %a = add i32 %x, 1
  %c = icmp eq i32 %x, 5
  br i1 %c, label %if.true, label %if.false
if.true:
  br i1 %arg, label %inner.a, label %inner.b
inner.a:
  br label %inner.merge
inner.b:
  br label %inner.merge
inner.merge:
  ret i32 %a
if.false:
  ret i32 0
}

; PHI operands are deliberately left untouched even inside the dominated
; region: rewriting them would need to account for which predecessor the
; value flows from. Use different incoming values so the PHI cannot be
; trivially simplified away before we get a chance to (not) rewrite it.
define i32 @phi_operand_not_touched(i32 %x, i1 %arg) {
; CHECK-LABEL: @phi_operand_not_touched(
; CHECK: inner.merge:
; CHECK-NEXT: %p = phi i32 [ %a, %inner.a ], [ 0, %inner.b ]
; CHECK-NEXT: ret i32 %p
  %a = add i32 %x, 1
  %c = icmp eq i32 %x, 5
  br i1 %c, label %if.true, label %if.false
if.true:
  br i1 %arg, label %inner.a, label %inner.b
inner.a:
  br label %inner.merge
inner.b:
  br label %inner.merge
inner.merge:
  %p = phi i32 [ %a, %inner.a ], [ 0, %inner.b ]
  ret i32 %p
if.false:
  ret i32 0
}

; llvm.fake.use operands are deliberately left untouched, but other uses in
; the same block are still folded normally.
define i32 @fake_use_operand_not_touched(i32 %x) {
; CHECK-LABEL: @fake_use_operand_not_touched(
; CHECK: if.true:
; CHECK-NEXT: call void (...) @llvm.fake.use(i32 %a)
; CHECK-NEXT: ret i32 6
  %a = add i32 %x, 1
  %c = icmp eq i32 %x, 5
  br i1 %c, label %if.true, label %if.false
if.true:
  call void (...) @llvm.fake.use(i32 %a)
  ret i32 %a
if.false:
  ret i32 0
}

; if.true is reachable from more than one predecessor, so the edge equality
; does not hold throughout it and no propagation should happen.
define i32 @not_only_reachable_via_edge(i32 %x, i1 %arg) {
; CHECK-LABEL: @not_only_reachable_via_edge(
; CHECK: if.true:
; CHECK-NEXT: ret i32 %a
  %a = add i32 %x, 1
  %c = icmp eq i32 %x, 5
  br i1 %c, label %if.true, label %other
other:
  br label %if.true
if.true:
  ret i32 %a
}

; Only integer equalities are supported: an expression built from a pointer
; known-equal to null should not be folded.
define i64 @pointer_type_not_propagated(ptr %x) {
; CHECK-LABEL: @pointer_type_not_propagated(
; CHECK: if.true:
; CHECK-NEXT: ret i64 %a
  %a = ptrtoint ptr %x to i64
  %c = icmp eq ptr %x, null
  br i1 %c, label %if.true, label %if.false
if.true:
  ret i64 %a
if.false:
  ret i64 0
}

; The dominated region (if.true) is nested inside a loop, and the other
; successor of the branch (if.false) can loop back around and reach if.true
; again on a later iteration. This should not prevent folding: whenever
; if.true executes, %x == 5 was just established by the icmp on that same
; iteration, regardless of how it was reached on prior iterations.
define i32 @loop_other_successor_can_reach_root(i32 %x, i32 %n) {
; CHECK-LABEL: @loop_other_successor_can_reach_root(
; CHECK: if.true:
; CHECK-NEXT: ret i32 6
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %if.false ]
  %a = add i32 %x, 1
  %c = icmp eq i32 %x, 5
  br i1 %c, label %if.true, label %if.false
if.true:
  ret i32 %a
if.false:
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret i32 0
}
