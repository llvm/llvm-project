; RUN: llc < %s -mtriple=avr | FileCheck %s

; CHECK: .globl __do_copy_data
@str = internal global [3 x i8] c"foo"

