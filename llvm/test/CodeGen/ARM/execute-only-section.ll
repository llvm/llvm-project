; RUN: llc < %s -mtriple=thumbv7m -mattr=+execute-only %s -o - | FileCheck %s
; RUN: llc < %s -mtriple=thumbv8m.base -mattr=+execute-only %s -o - | FileCheck %s
; RUN: llc < %s -mtriple=thumbv8m.main -mattr=+execute-only %s -o - | FileCheck %s

; CHECK:     .section .text,"axy",%progbits,unique,0
; CHECK-NOT: .section
; CHECK-NOT: .text
; CHECK:     .globl test_SectionForGlobal
; CHECK:     .type test_SectionForGlobal,%function
define void @test_SectionForGlobal() {
entry:
  ret void
}

; CHECK:     .section .test,"axy",%progbits
; CHECK-NOT: .section
; CHECK-NOT: .text
; CHECK:     .globl test_ExplicitSectionForGlobal
; CHECK:     .type test_ExplicitSectionForGlobal,%function
define void @test_ExplicitSectionForGlobal() section ".test" {
entry:
  ret void
}
