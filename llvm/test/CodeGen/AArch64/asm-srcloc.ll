; RUN: llc -O0 -stop-after=finalize-isel -o - %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; CHECK-LABEL: name: foo
; CHECK: INLINEASM {{.*}}, !0
define void @foo() {
  call void asm sideeffect "nowayisthisavalidinstruction", "r"(i32 0), !srcloc !0
  ret void
}

; CHECK-LABEL: name: bar
; CHECK: INLINEASM {{.*}}, !1
define void @bar() {
  call void asm sideeffect "nowayisthisavalidinstruction", ""(), !srcloc !1
  ret void
}

!0 = !{i32 23}
!1 = !{i32 91}
