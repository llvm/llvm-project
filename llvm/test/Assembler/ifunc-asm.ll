; RUN: llvm-as < %s | llvm-dis | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@foo = ifunc i32 (i32), i64 ()* @foo_ifunc
; CHECK: @foo = ifunc i32 (i32), i64 ()* @foo_ifunc

define internal i64 @foo_ifunc() {
entry:
  ret i64 0
}
; CHECK: define internal i64 @foo_ifunc()
