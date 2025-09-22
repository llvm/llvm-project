; RUN: opt -debug-only=branch-prob -passes=jump-threading -S %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Make sure that edges' probabilities would not accumulate if they are
; the same target BB.
; Edge L0 -> 2 and L0 -> 3 's targets are both L2, but their respective
; probability should not be L0 -> L2, because prob[L0->L2] equls to
; prob[L0->2] + prob[L0->3]

; CHECK: Computing probabilities for entry
; CHECK: eraseBlock L0
; CHECK-NOT: set edge L0 -> 0 successor probability to 0x12492492 / 0x80000000 = 14.29%
; CHECK-NOT: set edge L0 -> 1 successor probability to 0x24924925 / 0x80000000 = 28.57%
; CHECK-NOT: set edge L0 -> 2 successor probability to 0x24924925 / 0x80000000 = 28.57%
; CHECK-NOT: set edge L0 -> 3 successor probability to 0x24924925 / 0x80000000 = 28.57%
; CHECK: set edge L0 -> 0 successor probability to 0x1999999a / 0x80000000 = 20.00%
; CHECK: set edge L0 -> 1 successor probability to 0x33333333 / 0x80000000 = 40.00%
; CHECK: set edge L0 -> 2 successor probability to 0x1999999a / 0x80000000 = 20.00%
; CHECK: set edge L0 -> 3 successor probability to 0x1999999a / 0x80000000 = 20.00%
; CHECK-NOT: !0 = !{!"branch_weights", i32 306783378, i32 613566757, i32 613566757, i32 613566757}
; CHECK: !0 = !{!"branch_weights", i32 429496730, i32 858993459, i32 429496730, i32 429496730}
define void @test_switch(i1 %cond, i8 %value) nounwind {
entry:
  br i1 %cond, label %L0, label %L4
L0:
  %expr = select i1 %cond, i8 1, i8 %value
  switch i8 %expr, label %L3 [
    i8 1, label %L1
    i8 2, label %L2
    i8 3, label %L2
  ], !prof !0

L1:
  ret void
L2:
  ret void
L3:
  ret void
L4:
  br label %L0
}
!0 = !{!"branch_weights", i32 1, i32 7, i32 1, i32 1}
