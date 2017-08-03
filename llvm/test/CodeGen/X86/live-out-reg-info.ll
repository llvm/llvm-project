; RUN: llc < %s -march=x86-64 | grep testb

; Make sure dagcombine doesn't eliminate the comparison due
; to an off-by-one bug with computeKnownBits information.

declare void @qux()

define void @foo(i32 %a) {
  %t0 = lshr i32 %a, 23
  br label %next
next:
  %t1 = and i32 %t0, 256
  %t2 = icmp eq i32 %t1, 0
  br i1 %t2, label %true, label %false
true:
  call void @qux()
  ret void
false:
  ret void
}
