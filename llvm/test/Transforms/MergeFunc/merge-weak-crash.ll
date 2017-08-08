; RUN: opt -S -mergefunc < %s | FileCheck %s

; CHECK-LABEL: define i32 @func1
; CHECK: call i32 @func2
; CHECK: ret

; CHECK-LABEL: define i32 @func2
; CHECK: call i32 @unknown
; CHECK: ret

; CHECK-LABEL: define i32 @func4
; CHECK: call i32 @func2
; CHECK: ret

; CHECK-LABEL: define weak i32 @func3_weak
; CHECK: call i32 @func1
; CHECK: ret

define i32 @func1(i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  %sum2 = add i32 %sum, %y
  %sum3 = call i32 @func4(i32 %sum, i32 %sum2)
  ret i32 %sum3
}

define i32 @func4(i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  %sum2 = add i32 %sum, %y
  %sum3 = call i32 @unknown(i32 %sum, i32 %sum2)
  ret i32 %sum3
}

define weak i32 @func3_weak(i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  %sum2 = add i32 %sum, %y
  %sum3 = call i32 @func2(i32 %sum, i32 %sum2)
  ret i32 %sum3
}

define i32 @func2(i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  %sum2 = add i32 %sum, %y
  %sum3 = call i32 @unknown(i32 %sum, i32 %sum2)
  ret i32 %sum3
}

declare i32 @unknown(i32 %x, i32 %y)
