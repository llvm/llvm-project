; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@foo = dso_local ifunc i32 (i32), i32 (i32)* ()* @foo_ifunc
; CHECK: @foo = dso_local ifunc i32 (i32), i32 (i32)* ()* @foo_ifunc

define internal i32 (i32)* @foo_ifunc() {
entry:
  ret i32 (i32)* null
}
