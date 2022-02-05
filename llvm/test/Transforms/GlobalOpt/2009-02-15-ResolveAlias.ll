; RUN: opt < %s -passes=globalopt -S | FileCheck %s

define internal void @f() {
; CHECK-NOT: @f(
; CHECK: define dso_local void @a
	ret void
}

@a = dso_local alias void (), void ()* @f

define hidden void @g() {
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

