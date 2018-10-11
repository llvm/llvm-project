; RUN: llc -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s

; The load is to the high byte of the 2-byte store
@g = global i8 -75

define void @f(i16 %v) {
; CHECK-LABEL: f
; CHECK: sth 3, -2(1)
; CHECK: lbz 3, -2(1)
  %p32 = alloca i16
  store i16 %v, i16* %p32
  %p16 = bitcast i16* %p32 to i8*
  %tmp = load i8, i8* %p16
  store i8 %tmp, i8* @g
  ret void
}
