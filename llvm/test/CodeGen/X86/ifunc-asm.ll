; RUN: llvm-as < %s -o - | llc -filetype=asm | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define internal i64 @foo_ifunc() {
entry:
  ret i64 0
}
; CHECK: .type foo_ifunc,@function
; CHECK-NEXT: foo_ifunc:

@foo = ifunc i32 (i32), i64 ()* @foo_ifunc
; CHECK: .type foo,@function
; CHECK-NEXT: .type foo,@gnu_indirect_function
; CHECK-NEXT: foo = foo_ifunc
