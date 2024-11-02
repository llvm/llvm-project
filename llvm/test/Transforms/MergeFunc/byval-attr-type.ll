; RUN: opt -S -passes=mergefunc %s | FileCheck %s

@i = global i32 0
@f = global float 0.0

define internal void @foo() {
; CHECK: define internal void @foo()
  call void @callee_i32(i32* byval(i32) @i)
  ret void
}

define internal void @bar() {
; CHECK: define internal void @bar()
  call void @callee_float(float* byval(float) @f)
  ret void
}

define internal void @baz() {
; CHECK-NOT: define{{.*}}@bar
  call void @callee_float(float* byval(float) @f)
  ret void
}

define void @user() {
; CHECK-LABEL: define void @user
; CHECK: call void @foo()
; CHECK: call void @bar()
; CHECK: call void @bar()

  call void @foo()
  call void @bar()
  call void @baz()
  ret void
}

declare void @callee_i32(i32* byval(i32))
declare void @callee_float(float* byval(float))
