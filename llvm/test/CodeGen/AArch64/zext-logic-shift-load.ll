; RUN: llc -mtriple=aarch64-linux-gnu < %s -o - | FileCheck %s

define i32 @test1(i8* %p) {
; CHECK:       ldrb
; CHECK-NEXT:  ubfx
; CHECK-NEXT:  ret

  %1 = load i8, i8* %p
  %2 = lshr i8 %1, 1
  %3 = and i8 %2, 1
  %4 = zext i8 %3 to i32
  ret i32 %4
}

