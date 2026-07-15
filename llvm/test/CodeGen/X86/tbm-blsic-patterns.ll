; RUN: llc -mattr=+tbm < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: test_blsic_original
; CHECK: blsicl
define i32 @test_blsic_original(i32 %x) {
  %_2 = xor i32 %x, -1
  %_0.i = add i32 %x, -1
  %_0 = or i32 %_0.i, %_2
  ret i32 %_0
}

; CHECK-LABEL: test_blsic_optimized
; CHECK: blsicl
define i32 @test_blsic_optimized(i32 %x) {
  %1 = sub i32 0, %x
  %2 = and i32 %x, %1
  %3 = xor i32 %2, -1
  ret i32 %3
}

; 64-bit versions
; CHECK-LABEL: test_blsic_64_original
; CHECK: blsicq
define i64 @test_blsic_64_original(i64 %x) {
  %_2 = xor i64 %x, -1
  %_0.i = add i64 %x, -1
  %_0 = or i64 %_0.i, %_2
  ret i64 %_0
}

; CHECK-LABEL: test_blsic_64_optimized
; CHECK: blsicq
define i64 @test_blsic_64_optimized(i64 %x) {
  %1 = sub i64 0, %x
  %2 = and i64 %x, %1
  %3 = xor i64 %2, -1
  ret i64 %3
}
