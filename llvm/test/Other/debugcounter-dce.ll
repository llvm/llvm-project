; REQUIRES: asserts
; RUN: opt -passes=dce -S -debug-counter=dce-transform=1-2  < %s | FileCheck %s
;; Test that, with debug counters on, we will skip the first DCE opportunity, perform next 2,
;; and ignore all the others left.

; CHECK-LABEL: @test
; CHECK-NEXT: %add1 = add i32 1, 2
; CHECK-NEXT: %sub1 = sub i32 %add1, 1
; CHECK-NEXT: %add2 = add i32 1, 2
; CHECK-NEXT: %add3 = add i32 1, 2
; CHECK-NEXT: ret void
define void @test() {
  %add1 = add i32 1, 2
  %sub1 = sub i32 %add1, 1
  %add2 = add i32 1, 2
  %sub2 = sub i32 %add2, 1
  %add3 = add i32 1, 2
  %sub3 = sub i32 %add3, 1
  ret void
}
