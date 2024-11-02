; REQUIRES: x86-registered-target
; RUN: llc %s -o - | FileCheck %s --check-prefix=CHECK-SMALL

;; Check the llc option will override the IR input.
; RUN: llc -code-model=large %s -o - | FileCheck %s --check-prefix=CHECK-LARGE

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"Code Model", i32 1}

@data = internal constant [0 x i32] []

define ptr @foo() nounwind readonly {
entry:
; CHECK-LARGE: movabsq $data, %rax
; CHECK-SMALL: movl    $data, %eax
    ret ptr @data
}
