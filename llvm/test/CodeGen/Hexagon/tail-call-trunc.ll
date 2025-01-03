; RUN: llc -mtriple=hexagon < %s | FileCheck %s

declare i32 @ret_i32()

define i8 @test_i8() {
; CHECK-LABEL: test_i8:
; CHECK: jump ret_i32
  %res = tail call i32 @ret_i32()
  %val = trunc i32 %res to i8
  ret i8 %val
}

define i16 @test_i16() {
; CHECK-LABEL: test_i16:
; CHECK: jump ret_i32
  %res = tail call i32 @ret_i32()
  %val = trunc i32 %res to i16
  ret i16 %val
}

declare i64 @ret_i64()
define i32 @test_i32() {
; CHECK-LABEL: test_i32:
; CHECK: call ret_i64
  %res = tail call i64 @ret_i64()
  %val = trunc i64 %res to i32
  ret i32 42
}
