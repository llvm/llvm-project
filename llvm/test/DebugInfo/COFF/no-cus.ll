; RUN: llc < %s -filetype=obj -o %t.o
; RUN: llvm-objdump -section-headers %t.o | FileCheck %s

; Don't emit debug info in this scenario and don't crash.

; CHECK-NOT: .debug$S
; CHECK: .text
; CHECK-NOT: .debug$S

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.10.24728"

define void @f() {
entry:
  ret void
}

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"CodeView", i32 1}
!1 = !{i32 1, !"PIC Level", i32 2}
!2 = !{!"clang version 5.0.0 "}
