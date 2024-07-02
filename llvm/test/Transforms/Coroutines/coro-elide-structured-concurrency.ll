; Testing elide performed its job for calls to coroutines marked safe.
; RUN: opt < %s -S -passes='inline,coro-elide' | FileCheck %s

%struct.Task = type { ptr }

declare void @print(i32) nounwind

; resume part of the coroutine
define fastcc void @callee.resume(ptr dereferenceable(1)) {
  tail call void @print(i32 0)
  ret void
}

; destroy part of the coroutine
define fastcc void @callee.destroy(ptr) {
  tail call void @print(i32 1)
  ret void
}

; cleanup part of the coroutine
define fastcc void @callee.cleanup(ptr) {
  tail call void @print(i32 2)
  ret void
}

@callee.resumers = internal constant [3 x ptr] [
  ptr @callee.resume, ptr @callee.destroy, ptr @callee.cleanup]

declare void @alloc(i1) nounwind

; CHECK: define ptr @callee()
define ptr @callee() {
entry:
  %task = alloca %struct.Task, align 8
  %id = call token @llvm.coro.id(i32 0, ptr null,
                          ptr @callee,
                          ptr @callee.resumers)
  %alloc = call i1 @llvm.coro.alloc(token %id)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  store ptr %hdl, ptr %task
  ret ptr %task
}

; CHECK: define ptr @caller()
; Function Attrs: presplitcoroutine
define ptr @caller() #0 {
entry:
  %task = call ptr @callee()

  ; CHECK: %[[id:.+]] = call token @llvm.coro.id(i32 0, ptr null, ptr @callee, ptr @callee.resumers)
  ; CHECK-NOT: call i1 @llvm.coro.alloc(token %[[id]])
  call void @llvm.coro.safe.elide(ptr %task)

  ret ptr %task
}

attributes #0 = { presplitcoroutine }

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare ptr @llvm.coro.frame()
declare ptr @llvm.coro.subfn.addr(ptr, i8)
declare i1 @llvm.coro.alloc(token)
declare void @llvm.coro.safe.elide(ptr)
