; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_sllv(i32 %a, i32 %b) {
; CHECK: sllv $a0, $a0, $a1
; CHECK: SLLV_NM
  %sllv = shl i32 %a, %b
  ret i32 %sllv
}

define i32 @test_srlv(i32 %a, i32 %b) {
; CHECK: srlv $a0, $a0, $a1
; CHECK: SRLV_NM
  %sllv = lshr i32 %a, %b
  ret i32 %sllv
}
