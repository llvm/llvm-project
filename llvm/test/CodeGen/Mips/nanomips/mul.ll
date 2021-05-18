; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_mul(i32 %a, i32 %b) {
; CHECK: mul $a0, $a0, $a1
; CHECK: MUL_NM
  %mul = mul i32 %a, %b
  ret i32 %mul
}

define i64 @test_mul64(i64 %a, i64 %b) {
; CHECK: mul
; CHECK: MUL_NM
; CHECK: muhu
; CHECK: MUHU_NM
; CHECK: addu
; CHECK: ADDu_NM
; CHECK: mul
; CHECK: MUL_NM
; CHECK: addu
; CHECK: ADDu_NM
; CHECK: mul
; CHECK: MUL_NM
  %mul = mul i64 %a, %b
  ret i64 %mul
}

define i32 @test_mulhs(i32 %a, i32 %b) {
; CHECK: muh $a0, $a0, $a1
; CHECK: MUH_NM
  %1 = sext i32 %a to i64
  %2 = sext i32 %b to i64
  %3 = mul i64 %1, %2
  %4 = lshr i64 %3, 32
  %5 = trunc i64 %4 to i32
  ret i32 %5
}
