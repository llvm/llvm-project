; RUN: llc -mtriple=aarch64 < %s -o - | FileCheck %s

%struct.A = type { i8, i32 }
@foo = dso_local global %struct.A zeroinitializer, align 4

; CHECK:      .bss
; CHECK-NEXT: .globl foo
; CHECK-NEXT: .p2align 2, 0x0{{$}}
