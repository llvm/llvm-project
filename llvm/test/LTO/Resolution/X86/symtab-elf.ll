; RUN: llvm-as -o %t %s
; RUN: llvm-lto2 dump-symtab %t | FileCheck %s

; CHECK: target triple: x86_64-unknown-linux-gnu
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-NOT: linker opts:
!0 = !{i32 6, !"Linker Options", !{!{!"/include:foo"}}}
!llvm.module.flags = !{ !0 }

@g1 = global i32 0

; CHECK-NOT: fallback g1
@g2 = weak alias i32, i32* @g1
