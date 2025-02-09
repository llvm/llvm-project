; RUN: opt < %s -O0 -S -mtriple=wasm32-unknown-unknown | FileCheck %s
; REQUIRES: webassembly-registered-target

%swift.async_func_pointer = type <{ i32, i32 }>
@checkTu = global %swift.async_func_pointer <{ i32 ptrtoint (ptr @check to i32), i32 8 }>

define swiftcc void @check(ptr %0) {
entry:
  %1 = call token @llvm.coro.id.async(i32 0, i32 0, i32 0, ptr @checkTu)
  %2 = call ptr @llvm.coro.begin(token %1, ptr null)
  %3 = call ptr @llvm.coro.async.resume()
  store ptr %3, ptr %0, align 4
  %4 = call { ptr, i32 } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i32s(i32 0, ptr %3, ptr @__swift_async_resume_project_context, ptr @check.0, ptr null, ptr null)
  ret void
}

declare swiftcc void @check.0()
declare { ptr, i32 } @llvm.coro.suspend.async.sl_p0i32s(i32, ptr, ptr, ...)
declare token @llvm.coro.id.async(i32, i32, i32, ptr)
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare ptr @llvm.coro.async.resume()

define ptr @__swift_async_resume_project_context(ptr %0) {
entry:
  ret ptr null
}

; Verify that the resume call is not marked as musttail.
; CHECK-LABEL: define swiftcc void @check(
; CHECK-NOT: musttail call swiftcc void @check.0()
; CHECK:     call swiftcc void @check.0()
