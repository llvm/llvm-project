; RUN: llc -mtriple aarch64-windows -verify-machineinstrs -filetype asm -o - %s | FileCheck %s

define void @func(i64 %a) "no-stack-arg-probe" {
entry:
  %0 = alloca i8, i64 %a, align 16
  call void @func2(ptr nonnull %0)
  ret void
}

declare void @func2(ptr)

; CHECK: add [[REG1:x[0-9]+]], x0, #15
; CHECK-NOT: bl __chkstk
; CHECK: mov [[REG2:x[0-9]+]], sp
; CHECK: and [[REG1]], [[REG1]], #0xfffffffffffffff0
; CHECK: sub [[REG3:x[0-9]+]], [[REG2]], [[REG1]]
; CHECK: mov sp, [[REG3]]
