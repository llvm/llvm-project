; RUN: llc < %s -mtriple=armv6-apple-darwin -mcpu=arm1136jf-s | FileCheck %s
; Radar 8589805: Counting the number of microcoded operations, such as for an
; LDM instruction, was causing an assertion failure because the microop count
; was being treated as an instruction count.

; CHECK: push
; CHECK: pop
; CHECK: pop
; CHECK: pop

define i32 @test(i32 %x) {
entry:
  %0 = tail call signext i16 undef(i32* undef)
  switch i32 undef, label %bb3 [
    i32 0, label %bb4
    i32 1, label %bb1
    i32 2, label %bb2
  ]

bb1:
  ret i32 1

bb2:
  ret i32 2

bb3:
  ret i32 1

bb4:
  ret i32 3
}
