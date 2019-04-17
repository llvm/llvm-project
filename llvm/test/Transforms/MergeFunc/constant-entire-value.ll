; RUN: opt -S -mergefunc < %s | FileCheck %s

; RUN: opt -S -mergefunc < %s | FileCheck -check-prefix=NOPLUS %s

; This makes sure that zeros in constants don't cause problems with string based
; memory comparisons
define internal i32 @sum(i32 %x, i32 %y) {
; CHECK-LABEL: @sum
  %sum = add i32 %x, %y
  %1 = extractvalue [3 x i32] [ i32 3, i32 0, i32 2 ], 2
  %sum2 = add i32 %sum, %1
  %sum3 = add i32 %sum2, %y
  ret i32 %sum3
}

define internal i32 @add(i32 %x, i32 %y) {
; CHECK-LABEL: @add
  %sum = add i32 %x, %y
  %1 = extractvalue [3 x i32] [ i32 3, i32 0, i32 1 ], 2
  %sum2 = add i32 %sum, %1
  %sum3 = add i32 %sum2, %y
  ret i32 %sum3
}

define internal i32 @plus(i32 %x, i32 %y) {
; NOPLUS-NOT: @plus
  %sum = add i32 %x, %y
  %1 = extractvalue [3 x i32] [ i32 3, i32 0, i32 5 ], 2
  %sum2 = add i32 %sum, %1
  %sum3 = add i32 %sum2, %y
  ret i32 %sum3
}

define internal i32 @next(i32 %x, i32 %y) {
; CHECK-LABEL: @next
  %sum = add i32 %x, %y
  %1 = extractvalue [3 x i32] [ i32 3, i32 0, i32 5 ], 2
  %sum2 = add i32 %sum, %1
  %sum3 = add i32 %sum2, %y
  ret i32 %sum3
}

