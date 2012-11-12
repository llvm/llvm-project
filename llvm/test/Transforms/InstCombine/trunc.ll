; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; Instcombine should be able to eliminate all of these ext casts.

declare void @use(i32)

define i64 @test1(i64 %a) {
  %b = trunc i64 %a to i32
  %c = and i32 %b, 15
  %d = zext i32 %c to i64
  call void @use(i32 %b)
  ret i64 %d
; CHECK: @test1
; CHECK-NOT: ext
; CHECK: ret
}
define i64 @test2(i64 %a) {
  %b = trunc i64 %a to i32
  %c = shl i32 %b, 4
  %q = ashr i32 %c, 4
  %d = sext i32 %q to i64
  call void @use(i32 %b)
  ret i64 %d
; CHECK: @test2
; CHECK: shl i64 %a, 36
; CHECK: %d = ashr exact i64 {{.*}}, 36
; CHECK: ret i64 %d
}
define i64 @test3(i64 %a) {
  %b = trunc i64 %a to i32
  %c = and i32 %b, 8
  %d = zext i32 %c to i64
  call void @use(i32 %b)
  ret i64 %d
; CHECK: @test3
; CHECK-NOT: ext
; CHECK: ret
}
define i64 @test4(i64 %a) {
  %b = trunc i64 %a to i32
  %c = and i32 %b, 8
  %x = xor i32 %c, 8
  %d = zext i32 %x to i64
  call void @use(i32 %b)
  ret i64 %d
; CHECK: @test4
; CHECK: = and i64 %a, 8
; CHECK: = xor i64 {{.*}}, 8
; CHECK-NOT: ext
; CHECK: ret
}

define i32 @test5(i32 %A) {
  %B = zext i32 %A to i128
  %C = lshr i128 %B, 16
  %D = trunc i128 %C to i32
  ret i32 %D
; CHECK: @test5
; CHECK: %C = lshr i32 %A, 16
; CHECK: ret i32 %C
}

define i32 @test6(i64 %A) {
  %B = zext i64 %A to i128
  %C = lshr i128 %B, 32
  %D = trunc i128 %C to i32
  ret i32 %D
; CHECK: @test6
; CHECK: %C = lshr i64 %A, 32
; CHECK: %D = trunc i64 %C to i32
; CHECK: ret i32 %D
}

define i92 @test7(i64 %A) {
  %B = zext i64 %A to i128
  %C = lshr i128 %B, 32
  %D = trunc i128 %C to i92
  ret i92 %D
; CHECK: @test7
; CHECK: %B = zext i64 %A to i92
; CHECK: %C = lshr i92 %B, 32
; CHECK: ret i92 %C
}

define i64 @test8(i32 %A, i32 %B) {
  %tmp38 = zext i32 %A to i128
  %tmp32 = zext i32 %B to i128
  %tmp33 = shl i128 %tmp32, 32
  %ins35 = or i128 %tmp33, %tmp38
  %tmp42 = trunc i128 %ins35 to i64
  ret i64 %tmp42
; CHECK: @test8
; CHECK:   %tmp38 = zext i32 %A to i64
; CHECK:   %tmp32 = zext i32 %B to i64
; CHECK:   %tmp33 = shl nuw i64 %tmp32, 32
; CHECK:   %ins35 = or i64 %tmp33, %tmp38
; CHECK:   ret i64 %ins35
}

define i8 @test9(i32 %X) {
  %Y = and i32 %X, 42
  %Z = trunc i32 %Y to i8
  ret i8 %Z
; CHECK: @test9
; CHECK: trunc
; CHECK: and
; CHECK: ret
}

; rdar://8808586
define i8 @test10(i32 %X) {
  %Y = trunc i32 %X to i8
  %Z = and i8 %Y, 42
  ret i8 %Z
; CHECK: @test10
; CHECK: trunc
; CHECK: and
; CHECK: ret
}
