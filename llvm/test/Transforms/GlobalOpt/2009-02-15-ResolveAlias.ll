; RUN: opt < %s -passes=globalopt -S | FileCheck %s

define internal void @f() {
; CHECK-NOT: @f(
; CHECK: define void @a
	ret void
}

@a = alias void (), void ()* @f

define void @g() {
	call void() @a()
	ret void
}

@b = internal alias  void (),  void ()* @g
; CHECK-NOT: @b

define void @h() {
	call void() @b()
; CHECK: call void @g
	ret void
}

