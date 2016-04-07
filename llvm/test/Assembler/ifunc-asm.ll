; RUN: llvm-as < %s | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

target triple = "x86_64-unknown-linux-gnu"

@foo = ifunc i32 (i32), i64 ()* @foo_ifunc
; CHECK-LLVM: @foo = ifunc i32 (i32), i64 ()* @foo_ifunc

define internal i64 @foo_ifunc() {
entry:
  ret i64 0
}
; CHECK-LLVM: define internal i64 @foo_ifunc()
