; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_xor(i32 %a, i32 %b) {
; CHECK: xor $a0, $a0, $a1
; CHECK: XOR_NM
  %xor = xor i32 %a, %b
  ret i32 %xor
}

define i32 @test_ori0(i32 %a) {
; CHECK: xori $a0, $a0, 1
; CHECK: XORI_NM
  %xorred = xor i32 %a, 1
  ret i32 %xorred
}

define i32 @test_ori1(i32 %a) {
; CHECK: xori $a0, $a0, 4095
; CHECK: XORI_NM
  %xorred = xor i32 %a, 4095
  ret i32 %xorred
}

define i32 @test_ori2(i32 %a) {
; CHECK: li $a1, 4096
; CHECK: Li_NM
; CHECK: xor $a0, $a0, $a1
; CHECK: XOR_NM
  %xorred = xor i32 %a, 4096
  ret i32 %xorred
}
