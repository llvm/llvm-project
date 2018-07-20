; RUN: llc -mtriple=aarch64 < %s | FileCheck %s

; Check that we optimize out AND instructions and ADD/SUB instructions
; modulo the shift size to take advantage of the implicit mod done on
; the shift amount value by the variable shift/rotate instructions.

define i32 @test1(i32 %x, i64 %y) {
; CHECK-LABEL: test1:
; CHECK-NOT: and
; CHECK: lsr
  %sh_prom = trunc i64 %y to i32
  %shr = lshr i32 %x, %sh_prom
  ret i32 %shr
}

define i64 @test2(i32 %x, i64 %y) {
; CHECK-LABEL: test2:
; CHECK-NOT: orr
; CHECK-NOT: sub
; CHECK: neg
; CHECK: asr
  %sub9 = sub nsw i32 64, %x
  %sh_prom12.i = zext i32 %sub9 to i64
  %shr.i = ashr i64 %y, %sh_prom12.i
  ret i64 %shr.i
}

define i64 @test3(i64 %x, i64 %y) {
; CHECK-LABEL: test3:
; CHECK-NOT: add
; CHECK: lsl
  %add = add nsw i64 64, %x
  %shl = shl i64 %y, %add
  ret i64 %shl
}