; RUN: llc -mtriple x86_64-pc-linux-gnu < %s | FileCheck %s

$foo = comdat any
@bar = global i32 42, comdat($foo)
@foo = global i32 42

; CHECK:      .type   bar,@object
; CHECK-NEXT: .section        .data.bar,"aGw",@progbits,foo,comdat
; CHECK-NEXT: .globl  bar
; CHECK:      .type   foo,@object
; CHECK-NEXT: .data
; CHECK-NEXT: .globl  foo
