; RUN: llc -march=nvptx64 < %s | FileCheck %s

; Test for truncation from i64 to i32
define i32 @test_trunc_i64_to_i32(i64 %val) {
  ; CHECK-LABEL: test_trunc_i64_to_i32
  ; CHECK: trunc
  %trunc = trunc i64 %val to i32
  ret i32 %trunc
}

; Test for zero-extension from i32 to i64
define i64 @test_zext_i32_to_i64(i32 %val) {
  ; CHECK-LABEL: test_zext_i32_to_i64
  ; CHECK: zext
  %zext = zext i32 %val to i64
  ret i64 %zext
}