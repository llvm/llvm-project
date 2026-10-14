; REQUIRES: asserts
; RUN: llc < %s -O3 -mtriple=riscv64 -debug-only=stack-coloring 2>&1 | FileCheck %s

declare void @foo(...)

; CHECK: Slot #0 - 33179869176 bytes.
; CHECK: Slot #1 - 147483648 bytes.
; CHECK: Total Stack size: 33327352824 bytes
define dso_local void @stack_bigger_than_32bit_unsigned() {
entry:
  %a = alloca [4147483647 x i64], align 8
  %b = alloca [147483648 x i8], align 1
  %arraydecay = getelementptr inbounds [4147483647 x i64], ptr %a, i64 0, i64 0
  %arraydecay1 = getelementptr inbounds [147483648 x i8], ptr %b, i64 0, i64 0
  call void @foo(ptr noundef %arraydecay, ptr noundef %arraydecay1)
  ret void
}
