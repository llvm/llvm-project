; RUN: opt < %s -scalar-evolution -analyze | FileCheck %s

; CHECK-LABEL: @test1
; CHECK: -->  (zext
; CHECK: -->  (zext
; CHECK-NOT: -->  (zext

define i32 @test1(i32 %x) {
  %n = and i32 %x, 255
  %y = xor i32 %n, 255
  ret i32 %y
}

; ScalarEvolution shouldn't try to analyze %z into something like
;   -->  (zext i4 (-1 + (-1 * (trunc i64 (8 * %x) to i4))) to i64)
; or
;   -->  (8 * (zext i1 (trunc i64 ((8 * %x) /u 8) to i1) to i64))

; CHECK-LABEL: @test2
; CHECK: -->  (8 * (zext i1 (trunc i64 %x to i1) to i64))

define i64 @test2(i64 %x) {
  %a = shl i64 %x, 3
  %t = and i64 %a, 8
  %z = xor i64 %t, 8
  ret i64 %z
}
