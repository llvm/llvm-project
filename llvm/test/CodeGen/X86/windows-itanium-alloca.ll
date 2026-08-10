; RUN: llc -mtriple i686-windows-itanium -filetype asm -o - %s | FileCheck %s

target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"
target triple = "i686--windows-itanium"

declare void @external(ptr)

define dllexport void @alloca(i32 %sz) {
entry:
  %vla = alloca i8, i32 %sz, align 1
  call void @external(ptr %vla)
  ret void
}

; CHECK: __chkstk

