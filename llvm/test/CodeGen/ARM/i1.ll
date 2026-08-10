; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

define i32 @test1() {
; CHECK-LABEL: test1:
; CHECK:    mov r0, #0
; CHECK-NEXT:    cmp r0, #0
entry:
  br i1 undef, label %t, label %f

t:
  ret i32 4

f:
  ret i32 2
}
