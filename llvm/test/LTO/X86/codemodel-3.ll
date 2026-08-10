; RUN: llvm-as %s -o %t0.o
; RUN: llvm-as < %p/Inputs/codemodel-3.ll > %t1.o
; RUN: not llvm-lto2 run -r %t0.o,_start,px -r %t1.o,bar,px -r %t0.o,_GLOBAL_OFFSET_TABLE_, \
; RUN:   -r %t1.o,_GLOBAL_OFFSET_TABLE_, %t0.o %t1.o -o %t2.s 2>&1 | FileCheck %s 

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"Code Model", i32 4}

@data = internal constant [0 x i32] []

define ptr @_start() nounwind readonly {
entry:
    ret ptr @data
}

; CHECK: 'Code Model': IDs have conflicting values: 'i32 1' from {{.*}}, and 'i32 4' from {{.*}}
