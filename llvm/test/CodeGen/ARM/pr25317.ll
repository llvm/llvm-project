; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--linux-gnueabihf"

; CHECK-LABEL: f:
; CHECK: str lr, [r0]
define void @f(i32* %p) {
  call void asm sideeffect "str lr, $0", "=*o"(i32* %p)
  ret void
}
