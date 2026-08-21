; RUN: opt -passes=globalopt -S < %s | FileCheck %s

; CHECK: define internal fastcc i32 @foo(
define internal i32 @foo(i32 %x) noinline {
  ret i32 %x
}

; CHECK: define internal i32 @foo_optnone(
define internal i32 @foo_optnone(i32 %x) optnone noinline {
  ret i32 %x
}

define i32 @bar() {
  %r = call i32 @foo(i32 5)
  %s = call i32 @foo_optnone(i32 5)
  %res = add i32 %r, %s
  ret i32 %res
}
