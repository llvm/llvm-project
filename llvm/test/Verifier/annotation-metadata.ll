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

; CHECK: unsafe-stack-size annotation must have an integer value
; CHECK-NEXT: ptr @non_int_size
define void @non_int_size() !annotation !2 {
  ret void
}

; CHECK: unsafe-stack-size annotation must have an integer value
; CHECK-NEXT: ptr @missing_size
define void @missing_size() !annotation !3 {
  ret void
}

; CHECK-NOT: @int_size
define void @int_size() !annotation !4 {
  ret void
}

!2 = !{!"unsafe-stack-size", !"16"}
!3 = !{!"unsafe-stack-size"}
!4 = !{!"unsafe-stack-size", i32 16}
