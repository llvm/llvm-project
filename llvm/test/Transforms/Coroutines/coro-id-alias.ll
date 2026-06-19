; RUN: opt < %s -passes=coro-early -S | FileCheck %s

declare token @llvm.coro.id(i32, ptr, ptr, ptr)

@foo_alias = alias void (), ptr @foo

define void @foo() presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr @foo_alias, ptr null)
  ret void
}
; CHECK-LABEL: define void @foo()
; CHECK: call token @llvm.coro.id(i32 0, ptr null, ptr @foo, ptr null)
