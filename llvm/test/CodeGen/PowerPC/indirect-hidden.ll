; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu < %s | FileCheck %s

@a = external hidden global i32
@b = external global i32

define ptr @get_a() {
  ret ptr @a
}

define ptr @get_b() {
  ret ptr @b
}

; CHECK: .globl  get_a
; CHECK: .p2align 2
; CHECK: .type get_a,@function
; CHECK: .globl get_b
; CHECK: .p2align 2
; CHECK: .type get_b,@function
; CHECK: .hidden a
