; RUN: llc -mtriple=armv7-linux-gnu < %s -o - | FileCheck %s

define void @test1(i8* %p, i16* %q) {
; CHECK:       ldrb
; CHECK-NEXT:  mov
; CHECK-NEXT:  and
; CHECK-NEXT:  strh
; CHECK-NEXT:  bx

  %1 = load i8, i8* %p
  %2 = shl i8 %1, 2
  %3 = and i8 %2, 12
  %4 = zext i8 %3 to i16
  store i16 %4, i16* %q
  ret void
}

