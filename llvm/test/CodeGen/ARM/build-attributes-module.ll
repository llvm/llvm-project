; RUN: llc %s -mtriple=arm-none-none-eabi -o - | FileCheck %s

define void @f() {
    ret void
}

!llvm.module.flags = !{!0, !1, !2}

; CHECK-NOT: .eabi_attribute 20
!0 = !{i32 2, !"arm-eabi-fp-denormal", i32 0}
; CHECK: .eabi_attribute 21, 1
!1 = !{i32 2, !"arm-eabi-fp-exceptions", i32 1}
; CHECK: .eabi_attribute 23, 1
!2 = !{i32 2, !"arm-eabi-fp-number-model", i32 1}
