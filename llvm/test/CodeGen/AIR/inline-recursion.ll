; AIR has no call stack: a recursive function is an error, not an endless
; inlining loop (which ran llc out of memory).
; RUN: not llc -mtriple=air -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: AIR has no call stack: recursive function 'fact' cannot be compiled

define internal i32 @fact(i32 %n) {
entry:
  %z = icmp eq i32 %n, 0
  br i1 %z, label %one, label %rec
one:
  ret i32 1
rec:
  %m = sub i32 %n, 1
  %f = call i32 @fact(i32 %m)
  %r = mul i32 %n, %f
  ret i32 %r
}
define void @k(ptr addrspace(1) %p) "air-kernel" {
  %x = load i32, ptr addrspace(1) %p
  %y = call i32 @fact(i32 %x)
  store i32 %y, ptr addrspace(1) %p
  ret void
}
