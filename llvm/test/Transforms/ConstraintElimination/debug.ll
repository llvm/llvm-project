; RUN: opt -passes=constraint-elimination -S -debug %s 2>&1 | FileCheck %s

; REQUIRES: asserts

declare void @use(i1)

define i1 @test_and_ule(i4 %x, i4 %y, i4 %z) {
; CHECK:      Processing fact to add to the system:  %c.1 = icmp ule i4 %x, %y
; CHECK-NEXT: Adding 'ule %x, %y'
; CHECK-NEXT:  constraint: %x + -1 * %y <= 0

; CHECK:      Processing fact to add to the system:  %c.2 = icmp ule i4 %y, %z
; CHECK-NEXT: Adding 'ule %y, %z'
; CHECK-NEXT:  constraint: %y + -1 * %z <= 0

; CHECK: Checking   %t.1 = icmp ule i4 %x, %z
; CHECK: Condition   %t.1 = icmp ule i4 %x, %z implied by dominating constraints

; CHECK: Removing %x + -1 * %y <= 0
; CHECK: Removing %x + -1 * %y <= 0

entry:
  %c.1 = icmp ule i4 %x, %y
  %c.2 = icmp ule i4 %y, %z
  %and = and i1 %c.1, %c.2
  br i1 %and, label %bb1, label %exit

bb1:
  %t.1 = icmp ule i4 %x, %z
  ret i1 %t.1

exit:
  %c.3 = icmp ule i4 %x, %z
  ret i1 %c.3
}
