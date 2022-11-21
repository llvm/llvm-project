; REQUIRES: asserts
; RUN: opt -S -debug-counter=early-cse-skip=1,early-cse-count=1 -passes=early-cse -earlycse-debug-hash < %s 2>&1 | FileCheck %s
;; Test that, with debug counters on, we only optimize the second CSE opportunity.
define i32 @test(i32 %a, i32 %b) {
; CHECK-LABEL: @test(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    %add1 = add i32 %a, %b
; CHECK-NEXT:    %add2 = add i32 %a, %b
; CHECK-NEXT:    %add4 = add i32 %a, %b
; CHECK-NEXT:    %ret1 = add i32 %add1, %add2
; CHECK-NEXT:    %ret2 = add i32 %add1, %add4
; CHECK-NEXT:    %ret = add i32 %ret1, %ret2
; CHECK-NEXT:    ret i32 %ret
;
bb:
  %add1 = add i32 %a, %b
  %add2 = add i32 %a, %b
  %add3 = add i32 %a, %b
  %add4 = add i32 %a, %b
  %ret1 = add i32 %add1, %add2
  %ret2 = add i32 %add3, %add4
  %ret = add i32 %ret1, %ret2
  ret i32 %ret
}



