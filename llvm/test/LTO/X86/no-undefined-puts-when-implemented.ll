; RUN: llvm-as <%s >%t1
; RUN: llvm-lto -exported-symbol=_uses_puts -exported-symbol=_uses_printf -o - %t1 | \
; RUN: llvm-nm - | \
; RUN: FileCheck %s
; rdar://problem/16165191
; runtime library implementations should not be renamed

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin11"

@str = private unnamed_addr constant [13 x i8] c"hello world\0A\00"

; CHECK-NOT: U _puts
; CHECK: T _uses_printf
; CHECK: T _uses_puts
define i32 @uses_puts(i32 %i) {
entry:
  %s = call ptr @foo(i32 %i)
  %ret = call i32 @puts(ptr %s)
  ret i32 %ret
}
define i32 @uses_printf(i32 %i) {
entry:
  call i32 (ptr, ...) @printf(ptr @str)
  ret i32 0
}

define hidden i32 @printf(ptr readonly nocapture %fmt, ...) {
entry:
  %ret = call i32 @bar(ptr %fmt)
  ret i32 %ret
}
define hidden i32 @puts(ptr %s) {
entry:
  %ret = call i32 @bar(ptr %s)
  ret i32 %ret
}

declare ptr @foo(i32)
declare i32 @bar(ptr)
