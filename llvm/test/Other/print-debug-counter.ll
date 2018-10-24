; REQUIRES: asserts

; RUN: opt -S -debug-counter=early-cse-skip=1,early-cse-count=1 -early-cse \
; RUN:        -debug-counter=newgvn-vn-skip=1,newgvn-vn-count=2 -newgvn \
; RUN:        -instcombine -print-debug-counter < %s 2>&1 | FileCheck %s
;; Test debug counter prints correct info in right order.
; CHECK-LABEL: Counters and values:
; CHECK:       early-cse
; CHECK-SAME:  {4,1,1}
; CHECK:       instcombine-visit
; CHECK-SAME:  {12,0,-1}
; CHECK:       newgvn-vn
; CHECK-SAME:  {9,1,2}
define i32 @f1(i32 %a, i32 %b) {
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

define i32 @f2(i32 %a, i32 %b) {
bb:
  %add1 = add i32 %a, %b
  %add2 = add i32 %a, %b
  %ret = add i32 %add1, %add2
  ret i32 %ret
}
