; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

define weak i32 @sum(i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  %sum2 = add i32 %sum, %y
  %sum3 = add i32 %sum2, %y
  ret i32 %sum3
}

define weak i32 @add(i32 %x, i32 %y) {
  %sum = add i32 %x, %y
  %sum2 = add i32 %sum, %y
  %sum3 = add i32 %sum2, %y
  ret i32 %sum3
}

; Don't replace a weak function use by another equivalent function. We don't
; know whether the symbol that will ulitmately be linked is equivalent - we
; don't know that the weak definition is the definitive definition or whether it
; will be overriden by a stronger definition).

; CHECK-LABEL: define private i32 @0
; CHECK: add i32
; CHECK: add i32
; CHECK: add i32
; CHECK: ret

; CHECK-LABEL: define i32 @use_weak
; CHECK: call i32 @add
; CHECK: call i32 @sum
; CHECK: ret

; CHECK-LABEL: define weak i32 @sum
; CHECK:  tail call i32 @0
; CHECK:  ret

; CHECK-LABEL: define weak i32 @add
; CHECK:  tail call i32 @0
; CHECK:  ret


define i32 @use_weak(i32 %a, i32 %b) {
  %res = call i32 @add(i32 %a, i32 %b)
  %res2 = call i32 @sum(i32 %a, i32 %b)
  %res3 = add i32 %res, %res2
  ret i32 %res3
}
