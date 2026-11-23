; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Unsigned comparisons of short integers should not force a sign-extending
; load, since sign-extending the operands only makes the compared constant
; harder to encode.

; CHECK-LABEL: f0:
; CHECK-NOT: memh(
; CHECK: r[[R0:[0-9]+]] = memuh(r0+#4)
; CHECK: cmp.gtu(r[[R0]],#12)

define i32 @f0(ptr %p) nounwind {
entry:
  %a = getelementptr inbounds i16, ptr %p, i32 2
  %v = load i16, ptr %a, align 2
  %c = icmp ugt i16 %v, 12
  br i1 %c, label %exit0, label %exit1

exit0:
  ret i32 0

exit1:
  ret i32 1
}

; A constant with the sign bit of the short type set does not have to be
; materialized in a register.

; CHECK-LABEL: f1:
; CHECK-NOT: memh(
; CHECK: r[[R1:[0-9]+]] = memuh(r0+#0)
; CHECK: cmp.gtu(r[[R1]],##65523)

define i32 @f1(ptr %p) nounwind {
entry:
  %v = load i16, ptr %p, align 2
  %c = icmp ult i16 %v, 65524
  %r = zext i1 %c to i32
  ret i32 %r
}
