; RUN: llc -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s

; The load is to the high byte of the 2-byte store
@g = global i8 -75

define void @f(i16 %v) {
; CHECK-LABEL: f
; CHECK: sth 3, -2(1)
; CHECK: lbz 3, -2(1)
  %p32 = alloca i16
  store i16 %v, ptr %p32
  %tmp = load i8, ptr %p32
  store i8 %tmp, ptr @g
  ret void
}
