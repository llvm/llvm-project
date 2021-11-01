; RUN: llvm-as < %s | llvm-dis | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@foo = ifunc i32 (i32), i32 (i32)* ()* @foo_ifunc
; CHECK: @foo = ifunc i32 (i32), i32 (i32)* ()* @foo_ifunc

@strlen = ifunc i64 (i8*), bitcast (i64 (i32*)* ()* @mistyped_strlen_resolver to i64 (i8*)* ()*)
; CHECK: strlen = ifunc i64 (i8*), bitcast (i64 (i32*)* ()* @mistyped_strlen_resolver to i64 (i8*)* ()*)

define internal i32 (i32)* @foo_ifunc() {
entry:
  ret i32 (i32)* null
}
; CHECK: define internal i32 (i32)* @foo_ifunc()

define internal i64 (i32*)* @mistyped_strlen_resolver() {
entry:
  ret i64 (i32*)* null
}
; CHECK: define internal i64 (i32*)* @mistyped_strlen_resolver()
