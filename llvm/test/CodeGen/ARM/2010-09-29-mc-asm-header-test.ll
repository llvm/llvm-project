; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s
; This tests that MC/asm header conversion is smooth
;
; CHECK:      .syntax unified
; CHECK: .eabi_attribute 20, 1
; CHECK: .eabi_attribute 21, 1
; CHECK: .eabi_attribute 23, 3
; CHECK: .eabi_attribute 24, 1
; CHECK: .eabi_attribute 25, 1

define i32 @f(i64 %z) {
	ret i32 0
}
