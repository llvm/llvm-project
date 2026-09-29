; RUN: opt < %s "-passes=print<scalar-evolution>" -disable-output 2>&1 | FileCheck %s

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

; Check that we transform the naive lowering of the sequence below,
;   (4 * (zext i5 (2 * (trunc i32 %x to i5)) to i32)),
; to
;   (8 * (zext i4 (trunc i32 %x to i4) to i32))
;
; CHECK-LABEL: @test3
define i32 @test3(i32 %x) {
  %a = mul i32 %x, 8
; CHECK: %b
; CHECK-NEXT: --> (8 * (zext i4 (trunc i32 %x to i4) to i32))
  %b = and i32 %a, 124
  ret i32 %b
}
