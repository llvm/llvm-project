; RUN: llc -mtriple=thumb-unknown-unknown -no-integrated-as < %s | FileCheck %s --check-prefix=THUMB
; RUN: llc -mtriple=arm-unknown-unknown -no-integrated-as < %s | FileCheck %s --check-prefix=ARM

; In thumb mode, emit ".code 16" before global inline-asm instructions.

; THUMB: .code 16
; THUMB: stmib
; THUMB: .code 16

; ARM-NOT: .code 16
; ARM:     stmib

module asm "stmib sp, {r0-r14};"
