; RUN: opt < %s -inline -S | FileCheck %s

; Check that functions with "returns_twice" calls are only inlined,
; if they are themselve marked as such.

declare i32 @a() returns_twice

define i32 @inner1() {
entry:
  %call = call i32 @a() returns_twice
  %add = add nsw i32 1, %call
  ret i32 %add
}

define i32 @outer1() {
entry:
; CHECK-LABEL: define i32 @outer1(
; CHECK: call i32 @inner1()
  %call = call i32 @inner1()
  %add = add nsw i32 1, %call
  ret i32 %add
}

define i32 @inner2() returns_twice {
entry:
  %call = call i32 @a() returns_twice
  %add = add nsw i32 1, %call
  ret i32 %add
}

define i32 @outer2() {
entry:
; CHECK-LABEL: define i32 @outer2(
; CHECK: call i32 @a()
  %call = call i32 @inner2() returns_twice
  %add = add nsw i32 1, %call
  ret i32 %add
}

define i32 @inner3() {
entry:
  %invoke = invoke i32 @a() returns_twice
      to label %cont unwind label %lpad

cont:
  %add = add nsw i32 1, %invoke
  ret i32 %add

lpad:
  %lp = landingpad i32 personality i8* null cleanup
  resume i32 %lp
}

define i32 @outer3() {
entry:
; CHECK-LABEL: define i32 @outer3(
; CHECK: call i32 @inner3()
  %call = call i32 @inner3()
  %add = add nsw i32 1, %call
  ret i32 %add
}

define i32 @inner4() returns_twice {
entry:
  %invoke = invoke i32 @a() returns_twice
      to label %cont unwind label %lpad

cont:
  %add = add nsw i32 1, %invoke
  ret i32 %add

lpad:
  %lp = landingpad i32 personality i8* null cleanup
  resume i32 %lp
}

define i32 @outer4() {
entry:
; CHECK-LABEL: define i32 @outer4(
; CHECK: invoke i32 @a()
  %call = call i32 @inner4() returns_twice
  %add = add nsw i32 1, %call
  ret i32 %add
}
