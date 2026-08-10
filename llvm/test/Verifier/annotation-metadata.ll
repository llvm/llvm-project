; RUN: not llvm-as -disable-output < %s -o /dev/null 2>&1 | FileCheck %s

define void @test1(ptr %a) {
entry:
; CHECK: annotation must have at least one operand
  %a.addr = alloca ptr, align 8, !annotation !0

; CHECK-NEXT: operands must be a string or a tuple of strings
  ret void, !annotation !1
}

!0 = !{}
!1 = !{i32 10}
