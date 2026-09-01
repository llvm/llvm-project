; RUN: not llc -mtriple=x86_64-unknown-unknown < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: preserve_none calling convention does not support variadic functions

declare preserve_none void @vararg_func(i32, ...)

define void @test() {
  call preserve_none void (i32, ...) @vararg_func(i32 0)
  ret void
}
