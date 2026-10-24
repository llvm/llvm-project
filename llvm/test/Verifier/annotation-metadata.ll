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

; CHECK-NEXT: operands must be a string or a tuple of strings
define void @func_non_string() !annotation !1 {
  ret void
}

; CHECK-NEXT: unsafe-stack-size annotation must have an integer value
; CHECK-NEXT: !{!"unsafe-stack-size", !"16"}
define void @non_int_size() !annotation !2 {
  ret void
}

; CHECK-NEXT: unsafe-stack-size annotation must have an integer value
; CHECK-NEXT: !{!"unsafe-stack-size"}
define void @missing_size() !annotation !3 {
  ret void
}

; CHECK-NOT: {{.}}
define void @int_size() !annotation !4 {
  ret void
}

!2 = !{!"unsafe-stack-size", !"16"}
!3 = !{!"unsafe-stack-size"}
!4 = !{!"unsafe-stack-size", i32 16}
