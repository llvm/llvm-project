; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s
define void @test() {
entry:
  call void asm sideeffect "mtfsb1 ${0:s}", "i"(i32 7), !srcloc !1
  ret void
}
; CHECK: #APP
; CHECK-NEXT: mtfsb1 25

!1 = !{i32 40}
