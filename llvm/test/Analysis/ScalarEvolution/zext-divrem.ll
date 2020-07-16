; RUN: opt -analyze -enable-new-pm=0 -scalar-evolution -S < %s | FileCheck %s
; RUN: opt -disable-output "-passes=print<scalar-evolution>" -S < %s 2>&1 | FileCheck %s

define i64 @test1(i32 %a, i32 %b) {
; CHECK-LABEL: @test1
  %div = udiv i32 %a, %b
  %zext = zext i32 %div to i64
; CHECK: %zext
; CHECK-NEXT: -->  ((zext i32 %a to i64) /u (zext i32 %b to i64))
  ret i64 %zext
}

define i64 @test2(i32 %a, i32 %b) {
; CHECK-LABEL: @test2
  %rem = urem i32 %a, %b
  %zext = zext i32 %rem to i64
; CHECK: %zext
; CHECK-NEXT: -->  ((zext i32 %a to i64) + (-1 * (zext i32 %b to i64) * ((zext i32 %a to i64) /u (zext i32 %b to i64))))
  ret i64 %zext
}

define i64 @test3(i32 %a, i32 %b) {
; CHECK-LABEL: @test3
  %div = udiv i32 %a, %b
  %mul = mul i32 %div, %b
  %sub = sub i32 %a, %mul
  %zext = zext i32 %sub to i64
; CHECK: %zext
; CHECK-NEXT: -->  ((zext i32 %a to i64) + (-1 * (zext i32 %b to i64) * ((zext i32 %a to i64) /u (zext i32 %b to i64))))
  ret i64 %zext
}

define i64 @test4(i32 %t) {
; CHECK-LABEL: @test4
  %a = udiv i32 %t, 2
  %div = udiv i32 %t, 112
  %mul = mul i32 %div, 56
  %sub = sub i32 %a, %mul
  %zext = zext i32 %sub to i64
; CHECK: %zext
; CHECK-NEXT: -->  ((-56 * ((zext i32 %t to i64) /u 112)) + ((zext i32 %t to i64) /u 2))
  ret i64 %zext
}
