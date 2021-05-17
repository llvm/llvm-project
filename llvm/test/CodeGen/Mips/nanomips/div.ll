; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_div(i32 %a, i32 %b) {
; CHECK: div $a0, $a0, $a1
; CHECK: DIV_NM
; CHECK: teq $zero, $a1, 7
; CHECK: TEQ_NM
  %div = sdiv i32 %a, %b
  ret i32 %div
}

define i32 @test_mod(i32 %a, i32 %b) {
; CHECK: mod $a0, $a0, $a1
; CHECK: MOD_NM
; CHECK: teq $zero, $a1, 7
; CHECK: TEQ_NM
  %mod = srem i32 %a, %b
  ret i32 %mod
}

define i32 @test_divu(i32 %a, i32 %b) {
; CHECK: divu $a0, $a0, $a1
; CHECK: DIVU_NM
; CHECK: teq $zero, $a1, 7
; CHECK: TEQ_NM
  %div = udiv i32 %a, %b
  ret i32 %div
}

define i32 @test_modu(i32 %a, i32 %b) {
; CHECK: modu $a0, $a0, $a1
; CHECK: MODU_NM
; CHECK: teq $zero, $a1, 7
; CHECK: TEQ_NM
  %mod = urem i32 %a, %b
  ret i32 %mod
}

define i64 @test_div64(i64 %a, i64 %b) {
; CHECK: balc __divdi3
  %div = sdiv i64 %a, %b
  ret i64 %div
}

define i64 @test_mod64(i64 %a, i64 %b) {
; CHECK: balc __moddi3
  %mod = srem i64 %a, %b
  ret i64 %mod
}

define i64 @test_divu64(i64 %a, i64 %b) {
; CHECK: balc __udivdi3
  %div = udiv i64 %a, %b
  ret i64 %div
}

define i64 @test_modu64(i64 %a, i64 %b) {
; CHECK: balc __umoddi3
  %mod = urem i64 %a, %b
  ret i64 %mod
}
