; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

declare i32 @func0(i32)

define void @move.balc(i32 %a, i32 %b) {
; CHECK: move.balc $a0, $a1, func0
  call i32 @func0(i32 %b)
  ret void
}

declare i32 @func1(i32, i32, i32)

define void @move.balc1(i32 %a) {
; CHECK-NOT: move.balc $a2, $a0, func1
  %add1 = add i32 %a, 1
  %add2 = add i32 %a, 2
  call i32 @func1(i32 %add1, i32 %add2, i32 %a)
  ret void
}
