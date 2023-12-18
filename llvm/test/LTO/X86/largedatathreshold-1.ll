; RUN: llvm-as %s -o %t.o
; RUN: llvm-lto2 run -r %t.o,_start,px %t.o -o %t.s
; RUN: llvm-objdump -d %t.s.0 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@data = internal constant [20 x i8] zeroinitializer

define ptr @_start() {
entry:
; CHECK-LABEL:  <_start>:
; CHECK: leaq    (%rip), %rax
; CHECK-NOT: movabsq
    ret ptr @data
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"Code Model", i32 3}
!1 = !{i32 1, !"Large Data Threshold", i32 100}
