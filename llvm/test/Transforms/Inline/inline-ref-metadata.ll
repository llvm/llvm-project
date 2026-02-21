; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s

@a = global i32 1
@b = global i32 2
@c = global double 3.141593e+00

define i32 @callee1() !implicit.ref !0 {
  ret i32 0
}

define i32 @callee2() !implicit.ref !1 {
  ret i32 1
}

define i32 @callee3() {
  %i = call i32 @callee2()
  ret i32 %i
}
; CHECK: @callee3() !implicit.ref !1

define i32 @caller1() {
  %i = call i32 @callee1()
  ret i32 %i
}
; CHECK: @caller1() !implicit.ref !0

define i32 @caller2() !implicit.ref !2 {
  %i = call i32 @callee1()
  ret i32 %i
}
; CHECK: @caller2() !implicit.ref !2 !implicit.ref !0

define i32 @caller3() {
  %i = call i32 @caller4()
  ret i32 %i
}
; CHECK: @caller3() !implicit.ref !0 !implicit.ref !1

define i32 @caller4() {
  %a = call i32 @callee1()
  %b = call i32 @callee2()
  %add = add i32 %a, %b
  ret i32 %add
}
; CHECK: @caller4() !implicit.ref !0 !implicit.ref !1

!0 = !{ptr @a}
!1 = !{ptr @b}
!2 = !{ptr @c}

