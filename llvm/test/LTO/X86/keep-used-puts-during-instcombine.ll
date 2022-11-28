; RUN: opt -S -instcombine <%s | FileCheck %s
; rdar://problem/16165191
; llvm.compiler.used functions should not be renamed

target triple = "x86_64-apple-darwin11"

@llvm.compiler.used = appending global [1 x ptr] [
  ptr @puts
  ], section "llvm.metadata"
@llvm.used = appending global [1 x ptr] [
  ptr @uses_printf
  ], section "llvm.metadata"

@str = private unnamed_addr constant [13 x i8] c"hello world\0A\00"

define i32 @uses_printf(i32 %i) {
entry:
  call i32 (ptr, ...) @printf(ptr @str)
  ret i32 0
}

define internal i32 @printf(ptr readonly nocapture %fmt, ...) {
entry:
  %ret = call i32 @bar(ptr %fmt)
  ret i32 %ret
}

; CHECK: define {{.*}} @puts(
define internal i32 @puts(ptr %s) {
entry:
  %ret = call i32 @bar(ptr %s)
  ret i32 %ret
}

declare i32 @bar(ptr)
