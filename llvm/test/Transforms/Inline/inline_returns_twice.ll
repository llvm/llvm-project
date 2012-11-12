; RUN: opt < %s -inline -S | FileCheck %s

; Check that functions with "returns_twice" calls are only inlined,
; if they are themselve marked as such.

declare i32 @a() returns_twice
declare i32 @b() returns_twice

define i32 @f() {
entry:
  %call = call i32 @a() returns_twice
  %add = add nsw i32 1, %call
  ret i32 %add
}

define i32 @g() {
entry:
; CHECK: define i32 @g
; CHECK: call i32 @f()
; CHECK-NOT: call i32 @a()
  %call = call i32 @f()
  %add = add nsw i32 1, %call
  ret i32 %add
}

define i32 @h() returns_twice {
entry:
  %call = call i32 @b() returns_twice
  %add = add nsw i32 1, %call
  ret i32 %add
}

define i32 @i() {
entry:
; CHECK: define i32 @i
; CHECK: call i32 @b()
; CHECK-NOT: call i32 @h()
  %call = call i32 @h() returns_twice
  %add = add nsw i32 1, %call
  ret i32 %add
}
