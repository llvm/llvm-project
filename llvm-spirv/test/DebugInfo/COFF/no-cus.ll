; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=%triple < %t.ll -filetype=obj -o %t.o
; RUN: llvm-objdump -section-headers %t.o | FileCheck %s

; Don't emit debug info in this scenario and don't crash.

; CHECK-NOT: .debug$S
; CHECK: .text
; CHECK-NOT: .debug$S

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define void @f() {
entry:
  ret void
}

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"CodeView", i32 1}
!1 = !{i32 1, !"PIC Level", i32 2}
!2 = !{!"clang version 5.0.0 "}
