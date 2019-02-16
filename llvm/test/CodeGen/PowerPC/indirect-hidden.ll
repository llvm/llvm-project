; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu < %s | FileCheck %s

@a = external hidden global i32
@b = external global i32

define i32* @get_a() {
  ret i32* @a
}

define i32* @get_b() {
  ret i32* @b
}

; CHECK: .globl  get_a
; CHECK: .p2align 2
; CHECK: .type get_a,@function
; CHECK: .globl get_b
; CHECK: .p2align 2
; CHECK: .type get_b,@function
; CHECK: .hidden a
