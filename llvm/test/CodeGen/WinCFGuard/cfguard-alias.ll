; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s

; CHECK: .section .gfids$y
; CHECK: .symidx alias


define void @func() {
  ret void
}

@alias = alias ptr, ptr @func

@ptrs = global [1 x ptr] [ptr @alias]

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}
