; RUN: opt -S -reassociate < %s | FileCheck %s
; rdar://8944681

; Reassociate should clear optional flags like nsw when reassociating.

; CHECK: @test0
; CHECK: %y = add i64 %b, %a
; CHECK: %z = add i64 %y, %c
define i64 @test0(i64 %a, i64 %b, i64 %c) {
  %y = add nsw i64 %c, %b
  %z = add i64 %y, %a
  ret i64 %z
}

; CHECK: @test1
; CHECK: %y = add i64 %b, %a
; CHECK: %z = add i64 %y, %c
define i64 @test1(i64 %a, i64 %b, i64 %c) {
  %y = add i64 %c, %b
  %z = add nsw i64 %y, %a
  ret i64 %z
}

; PR9215
; CHECK: %s = add nsw i32 %y, %x
define i32 @test2(i32 %x, i32 %y) {
  %s = add nsw i32 %x, %y
  ret i32 %s
}
