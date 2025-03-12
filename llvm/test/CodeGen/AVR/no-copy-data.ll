; RUN: llc < %s -mtriple=avr | FileCheck %s

; CHECK-NOT: .globl __do_copy_data
; CHECK:     .globl __do_clear_bss

@noinit = internal global i8 5, section ".noinit"
@external = external global i8
@global = global i8 0
