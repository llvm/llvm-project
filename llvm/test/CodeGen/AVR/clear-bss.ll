; RUN: llc < %s -march=avr | FileCheck %s

; CHECK: .globl __do_clear_bss
@zeroed = internal constant [3 x i8] zeroinitializer

