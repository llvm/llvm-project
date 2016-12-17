; RUN: llc -mtriple arm-unknown -global-isel %s -o - | FileCheck %s

define void @test_void_return() {
; CHECK-LABEL: test_void_return:
; CHECK: bx lr
entry:
  ret void
}

define i32 @test_add(i32 %x, i32 %y) {
; CHECK-LABEL: test_add:
; CHECK: add r0, r0, r1
; CHECK: bx lr
entry:
  %sum = add i32 %x, %y
  ret i32 %sum
}
