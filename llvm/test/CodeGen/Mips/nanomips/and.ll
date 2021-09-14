; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @test_and(i32 %a, i32 %b) {
; CHECK: and $a0, $a0, $a1
; CHECK: AND_NM
  %anded = and i32 %a, %b
  ret i32 %anded
}

define i32 @test_andi0(i32 %a) {
; CHECK: andi $a0, $a0, 1
; CHECK: ANDI_NM
  %anded = and i32 %a, 1
  ret i32 %anded
}

define i32 @test_andi1(i32 %a) {
; CHECK: andi $a0, $a0, 4095
; CHECK: ANDI_NM
  %anded = and i32 %a, 4095
  ret i32 %anded
}

define i32 @test_andi2(i32 %a) {
; CHECK: li $t4, 4096
; CHECK: Li_NM
; CHECK: and $a0, $a0, $t4
; CHECK: AND_NM
  %anded = and i32 %a, 4096
  ret i32 %anded
}
