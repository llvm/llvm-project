; RUN: llc -mtriple=aarch64 -mattr=+execute-only %s -o - | FileCheck %s

$test_comdat = comdat any

; CHECK:     .section .text,"axy",@progbits,unique,0
; CHECK-NOT: .section
; CHECK-NOT: .text
; CHECK:     .globl test_section_for_global
; CHECK:     .type test_section_for_global,@function
define void @test_section_for_global() {
entry:
  ret void
}

; CHECK:     .section .text.test_comdat,"axGy",@progbits,test_comdat,comdat,unique,0
; CHECK-NOT: .section
; CHECK-NOT: .text
; CHECK:     .weak test_comdat
; CHECK:     .type test_comdat,@function
define linkonce_odr void @test_comdat() comdat {
entry:
  ret void
}

; CHECK:     .section .test,"axy",@progbits
; CHECK-NOT: .section
; CHECK-NOT: .text
; CHECK:     .globl test_explicit_section_for_global
; CHECK:     .type test_explicit_section_for_global,@function
define void @test_explicit_section_for_global() section ".test" {
entry:
  ret void
}

; CHECK:     .rodata,"a",@progbits
; CHECK-NOT: .section
; CHECK-NOT: .text
; CHECK:     .globl test_rodata
@test_rodata = constant i32 0, align 4
