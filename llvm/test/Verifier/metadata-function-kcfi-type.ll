; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define void @a() {
  unreachable
}

define void @b() !kcfi_type !0 {
  unreachable
}

; CHECK: function must have a single !kcfi_type attachment
define void @f0() !kcfi_type !0 !kcfi_type !0 {
  unreachable
}
!0 = !{i32 10}

; CHECK: !kcfi_type must have exactly one operand
define void @f1() !kcfi_type !1 {
  unreachable
}
!1 = !{!"string", i32 0}

; CHECK: expected a constant operand for !kcfi_type
define void @f2() !kcfi_type !2 {
  unreachable
}
!2 = !{!"string"}

; CHECK: expected a constant integer operand for !kcfi_type
define void @f3() !kcfi_type !3 {
  unreachable
}
!3 = !{ptr @f3}

; CHECK: expected a 32-bit integer constant operand for !kcfi_type
define void @f4() !kcfi_type !4 {
  unreachable
}
!4 = !{i64 10}
