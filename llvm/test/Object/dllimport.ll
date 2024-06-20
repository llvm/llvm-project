; RUN: llvm-as %s -o - | llvm-nm - | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; CHECK: U __imp_f
; CHECK: U __imp_v
; CHECK: T g

declare dllimport void @f()
@v = external dllimport global i32

define void @g() {
  call void @f()
  store i32 42, ptr @v
  ret void
}
