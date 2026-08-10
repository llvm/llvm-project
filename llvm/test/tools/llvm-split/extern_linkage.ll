; Test that extern_weak linkage is preserved.
; RUN: llvm-split -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck %s
; RUN: llvm-dis -o - %t1 | FileCheck %s

; Both declarations are extern_weak in all partitions.

; CHECK: @x = extern_weak global i32, align 4
@x = extern_weak global i32, align 4

; CHECK: declare extern_weak void @f(...)
declare extern_weak void @f(...)
