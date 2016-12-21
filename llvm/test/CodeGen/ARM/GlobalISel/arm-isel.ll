; RUN: llc -mtriple arm-unknown -global-isel %s -o - | FileCheck %s

define void @test_void_return() {
; CHECK-LABEL: test_void_return:
; CHECK: bx lr
entry:
  ret void
}

define i8 @test_add_i8(i8 %x, i8 %y) {
; CHECK-LABEL: test_add_i8:
; CHECK: add r0, r0, r1
; CHECK: bx lr
entry:
  %sum = add i8 %x, %y
  ret i8 %sum
}

define i16 @test_add_i16(i16 %x, i16 %y) {
; CHECK-LABEL: test_add_i16:
; CHECK: add r0, r0, r1
; CHECK: bx lr
entry:
  %sum = add i16 %x, %y
  ret i16 %sum
}

define i32 @test_add_i32(i32 %x, i32 %y) {
; CHECK-LABEL: test_add_i32:
; CHECK: add r0, r0, r1
; CHECK: bx lr
entry:
  %sum = add i32 %x, %y
  ret i32 %sum
}

define i32 @test_many_args(i32 %p0, i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5) {
; CHECK-LABEL: test_many_args:
; CHECK: add [[P5ADDR:r[0-9]+]], sp, #4
; CHECK: ldr [[P5:r[0-9]+]], {{.*}}[[P5ADDR]]
; CHECK: add r0, r2, [[P5]]
; CHECK: bx lr
entry:
  %sum = add i32 %p2, %p5
  ret i32 %sum
}
