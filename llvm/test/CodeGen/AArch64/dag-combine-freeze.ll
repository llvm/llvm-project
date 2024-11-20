; RUN: llc -mtriple aarch64 -o /dev/null %s

; This used to fail with:
;    Assertion `N1.getOpcode() != ISD::DELETED_NODE &&
;               "Operand is DELETED_NODE!"' failed.
; Just make sure we do not crash here.
define void @test_fold_freeze_over_select_cc(i15 %a, ptr %p1, ptr %p2) {
entry:
  %a2 = add nsw i15 %a, 1
  %sext = sext i15 %a2 to i32
  %ashr = ashr i32 %sext, 31
  %lshr = lshr i32 %ashr, 7
  ; Setup an already frozen input to ctlz.
  %freeze = freeze i32 %lshr
  %ctlz = call i32 @llvm.ctlz.i32(i32 %freeze, i1 true)
  store i32 %ctlz, ptr %p1, align 1
  ; Here is another ctlz, which is used by a frozen select.
  ; DAGCombiner::visitFREEZE will to try to fold the freeze over a SELECT_CC,
  ; and when dealing with the condition operand the other SELECT_CC operands
  ; will be replaced/simplified as well. So the SELECT_CC is mutated while
  ; freezing the "maybe poison operands". This needs to be handled by
  ; DAGCombiner::visitFREEZE, as it can't store the list of SDValues that
  ; should be frozen in a separate data structure that isn't updated when the
  ; SELECT_CC is mutated.
  %ctlz1 = call i32 @llvm.ctlz.i32(i32 %lshr, i1 true)
  %icmp = icmp ne i32 %lshr, 0
  %select = select i1 %icmp, i32 %ctlz1, i32 0
  %freeze1 = freeze i32 %select
  store i32 %freeze1, ptr %p2, align 1
  ret void
}
