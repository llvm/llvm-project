; REQUIRES: x86-registered-target
; RUN: opt -o %t %s
; RUN: not llvm-lto2 dump-symtab %t 2>&1 | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "x86_64-pc-windows-msvc"

@bar = global i32 0

; CHECK: Invalid weak external
@foo = weak alias i32, i32* getelementptr (i32, i32* @bar, i32 1)
