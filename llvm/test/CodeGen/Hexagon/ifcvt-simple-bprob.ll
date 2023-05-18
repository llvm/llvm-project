; RUN: llc -march=hexagon < %s

; Check that branch probabilities are set correctly after performing the
; simple variant of if-conversion. The converted block has a branch that
; is not analyzable.

target triple = "hexagon"

declare void @foo()

; CHECK-LABEL: danny
; CHECK: if (p0.new) jump:nt foo
define void @danny(i32 %x) {
  %t0 = icmp sgt i32 %x, 0
  br i1 %t0, label %tail, label %exit, !prof !0
tail:
  tail call void @foo();
  ret void
exit:
  ret void
}

; CHECK-LABEL: sammy
; CHECK: if (!p0.new) jump:t foo
define void @sammy(i32 %x) {
  %t0 = icmp sgt i32 %x, 0
  br i1 %t0, label %exit, label %tail, !prof !0
tail:
  tail call void @foo();
  ret void
exit:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 2000}

