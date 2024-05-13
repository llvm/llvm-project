; RUN: opt < %s -passes=inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s

define internal i32 @callee1(i32 %A, i32 %B) {
; CHECK-NOT: @callee1
entry:
  %cond = icmp eq i32 %A, 123
  br i1 %cond, label %T, label %F

T:
  %C = mul i32 %B, %B
  ret i32 %C

F:
  ret i32 0
}

define internal i32 @callee2(i32 %A, i32 %B) {
; CHECK-NOT: @callee2
entry:
  switch i32 %A, label %T [
           i32 10, label %F
           i32 1234, label %G
  ]

dead:
  %cond = icmp eq i32 %A, 123
  br i1 %cond, label %T, label %F

T:
  %C = mul i32 %B, %B
  ret i32 %C

F:
  ret i32 0

G:
  %D = mul i32 %B, %B
  %E = mul i32 %D, %B
  ret i32 %E
}

define i32 @test(i32 %A) {
; CHECK-LABEL: define i32 @test(i32 %A)
entry:
  %X = call i32 @callee1( i32 10, i32 %A )
  %Y = call i32 @callee2( i32 10, i32 %A )
; CHECK-NOT: call
; CHECK-NOT: mul

  %Z = add i32 %X, %Y
  ret i32 %Z
}

