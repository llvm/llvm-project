; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_sllv(i32 %a, i32 %b) {
; CHECK: sllv $a0, $a0, $a1
; CHECK: SLLV_NM
  %sllv = shl i32 %a, %b
  ret i32 %sllv
}

define i32 @test_sll(i32 %a) {
; CHECK: sll $a0, $a0, 10
; CHECK: SLL_NM
  %sll = shl i32 %a, 10
  ret i32 %sll
}

define i32 @test_srlv(i32 %a, i32 %b) {
; CHECK: srlv $a0, $a0, $a1
; CHECK: SRLV_NM
  %srlv = lshr i32 %a, %b
  ret i32 %srlv
}

define i32 @test_srl(i32 %a) {
; CHECK: srl $a0, $a0, 10
; CHECK: SRL_NM
  %srl = lshr i32 %a, 10
  ret i32 %srl
}

define i32 @test_srav(i32 %a, i32 %b) {
; CHECK: srav $a0, $a0, $a1
; CHECK: SRAV_NM
  %srav = ashr i32 %a, %b
  ret i32 %srav
}

define i32 @test_sra(i32 %a) {
; CHECK: sra $a0, $a0, 10
; CHECK: SRA_NM
  %sra = ashr i32 %a, 10
  ret i32 %sra
}

define i32 @test_rotrv(i32 %a, i32 %b) {
; CHECK-NOT: srlv
; CHECK-NOT: sllv
; CHECK: rotrv $a0, $a0, $a1
; CHECK: ROTRV_NM
  %sub = sub i32 32, %b
  %shl = shl i32 %a, %sub
  %shr = lshr i32 %a, %b
  %or = or i32 %shl, %shr
  ret i32 %or
}
