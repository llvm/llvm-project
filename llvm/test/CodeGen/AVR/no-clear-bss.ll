; RUN: llc < %s -mtriple=avr | FileCheck %s

; CHECK:     .globl __do_copy_data
; CHECK-NOT: .globl __do_clear_bss

@str = internal global [3 x i8] c"foo"
@noinit = internal constant [3 x i8] zeroinitializer, section ".noinit"
@external = external constant [3 x i8]
