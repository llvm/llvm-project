; RUN: llvm-as %s -o - | llvm-nm - | FileCheck %s

; We should technically emit an unmangled reference to f here,
; but no existing linker needs this.

; XFAIL: *

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; CHECK: U f

declare dllimport void @f()
@fp = constant ptr @f
