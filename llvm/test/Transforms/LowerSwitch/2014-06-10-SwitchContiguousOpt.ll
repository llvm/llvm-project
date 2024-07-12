; RUN: opt < %s -passes=lower-switch -S | FileCheck %s
; CHECK-NOT: icmp eq i32 %0, 1

define i32 @foo(i32 %a) #0 {
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  switch i32 %0, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
  ]

sw.bb:
  ret i32 12

sw.bb1:
  ret i32 4

sw.bb2:
  ret i32 2

sw.default:
  ret i32 9
}
