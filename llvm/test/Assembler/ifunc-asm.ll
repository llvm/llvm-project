; RUN: llvm-as  < %s | llvm-dis  | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@foo = ifunc i32 (i32), ptr @foo_ifunc
; CHECK: @foo = ifunc i32 (i32), ptr @foo_ifunc

@strlen = ifunc i64 (ptr), ptr @mistyped_strlen_resolver
; CHECK: strlen = ifunc i64 (ptr), ptr @mistyped_strlen_resolver

define internal ptr @foo_ifunc() {
entry:
  ret ptr null
}
; CHECK: define internal ptr @foo_ifunc()

define internal ptr @mistyped_strlen_resolver() {
entry:
  ret ptr null
}
; CHECK: define internal ptr @mistyped_strlen_resolver()
