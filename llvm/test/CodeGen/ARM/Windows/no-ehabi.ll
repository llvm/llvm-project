; RUN: llc -mtriple=thumbv7-windows -mcpu=cortex-a9 -o - %s | FileCheck %s

declare void @callee(i32 %i)

define i32 @caller(i32 %i, i32 %j, i32 %k, i32 %l, i32 %m, i32 %n, i32 %o,
                   i32 %p) {
entry:
  %q = add nsw i32 %j, %i
  %r = add nsw i32 %q, %k
  %s = add nsw i32 %r, %l
  call void @callee(i32 %s)
  %t = add nsw i32 %n, %m
  %u = add nsw i32 %t, %o
  %v = add nsw i32 %u, %p
  call void @callee(i32 %v)
  %w = add nsw i32 %v, %s
  ret i32 %w
}

; CHECK-NOT: .save {{{.*}}}

