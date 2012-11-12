; RUN: opt < %s -globalopt -S | FileCheck %s

define internal void @f() {
; CHECK-NOT: @f
; CHECK: define void @a
	ret void
}

@a = alias void ()* @f

define void @g() {
	call void()* @a()
	ret void
}

@b = alias internal void ()* @g
; CHECK-NOT: @b

define void @h() {
	call void()* @b()
; CHECK: call void @g
	ret void
}

