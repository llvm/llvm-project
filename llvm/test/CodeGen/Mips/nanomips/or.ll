; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_or(i32 %a, i32 %b) {
; CHECK: or $a0, $a0, $a1
; CHECK: OR_NM
  %orred = or i32 %a, %b
  ret i32 %orred
}

define i32 @test_ori0(i32 %a) {
; CHECK: ori $a0, $a0, 1
; CHECK: ORI_NM
  %orred = or i32 %a, 1
  ret i32 %orred
}

define i32 @test_ori1(i32 %a) {
; CHECK: ori $a0, $a0, 4095
; CHECK: ORI_NM
  %orred = or i32 %a, 4095
  ret i32 %orred
}

define i32 @test_ori2(i32 %a) {
; CHECK: li $t4, 4096
; CHECK: Li_NM
; CHECK: or $a0, $a0, $t4
; CHECK: OR_NM
  %orred = or i32 %a, 4096
  ret i32 %orred
}
