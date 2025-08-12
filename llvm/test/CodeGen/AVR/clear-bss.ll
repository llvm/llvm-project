; RUN: llc < %s -mtriple=avr | FileCheck %s

; CHECK: .globl __do_clear_bss
@zeroed = internal global [3 x i8] zeroinitializer
@common = common global i8 0
