; Tests that the coro.destroy and coro.resume are devirtualized where possible,
; SCC pipeline restarts and inlines the direct calls.
; RUN: opt < %s -S \
; RUN: -passes='cgscc(repeat<2>(inline,function(coro-elide,dce)))' \
; RUN:   | FileCheck %s

declare void @print(i32) nounwind

; resume part of the coroutine
define fastcc void @f.resume(ptr dereferenceable(1)) {
  tail call void @print(i32 0)
  ret void
}

; destroy part of the coroutine
define fastcc void @f.destroy(ptr) {
  tail call void @print(i32 1)
  ret void
}

; cleanup part of the coroutine
define fastcc void @f.cleanup(ptr) {
  tail call void @print(i32 2)
  ret void
}

@f.resumers = internal constant [3 x ptr] [ptr @f.resume,
                                                   ptr @f.destroy,
                                                   ptr @f.cleanup]

; a coroutine start function
define ptr @f() {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null,
                          ptr @f,
                          ptr @f.resumers)
  %alloc = call i1 @llvm.coro.alloc(token %id)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  ret ptr %hdl
}

; CHECK-LABEL: @callResume(
define void @callResume() {
entry:
  %hdl = call ptr @f()

; CHECK: call void @print(i32 0)
  %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %0(ptr %hdl)

; CHECK-NEXT: call void @print(i32 2)
  %1 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %1(ptr %hdl)

; CHECK-NEXT: ret void
  ret void
}

; CHECK-LABEL: @callResumeMultiRet(
define void @callResumeMultiRet(i1 %b) {
entry:
  %hdl = call ptr @f()
; CHECK: %alloc.i = call i1 @llvm.coro.alloc
; CHECK: call void @print(i32 0)
  %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %0(ptr %hdl)
  br i1 %b, label %destroy, label %ret

destroy:
; CHECK: call void @print(i32 1)
  %1 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %1(ptr %hdl)
  ret void

ret:
  ret void
}

; CHECK-LABEL: @callResumeMultiRetDommmed(
define void @callResumeMultiRetDommmed(i1 %b) {
entry:
  %hdl = call ptr @f()
; CHECK-NOT: %alloc.i = call i1 @llvm.coro.alloc
; CHECK: call void @print(i32 0)
  %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %0(ptr %hdl)
; CHECK: call void @print(i32 2)
  %1 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %1(ptr %hdl)
  br i1 %b, label %destroy, label %ret

destroy:
  ret void

ret:
  ret void
}

; CHECK-LABEL: @eh(
define void @eh() personality ptr null {
entry:
  %hdl = call ptr @f()

; CHECK: call void @print(i32 0)
  %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  invoke void %0(ptr %hdl)
          to label %cont unwind label %ehcleanup
cont:
  ret void

ehcleanup:
  %tok = cleanuppad within none []
  cleanupret from %tok unwind to caller
}

; CHECK-LABEL: @no_devirt_info_null(
; no devirtualization here, since coro.begin info parameter is null
define void @no_devirt_info_null() {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)

; CHECK: call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %0(ptr %hdl)

; CHECK: call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  %1 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %1(ptr %hdl)

; CHECK: ret void
  ret void
}

; CHECK-LABEL: @no_devirt_no_begin(
; no devirtualization here, since coro.begin is not visible
define void @no_devirt_no_begin(ptr %hdl) {
entry:

; CHECK: call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %0(ptr %hdl)

; CHECK: call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  %1 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %1(ptr %hdl)

; CHECK: ret void
  ret void
}

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare ptr @llvm.coro.frame()
declare ptr @llvm.coro.subfn.addr(ptr, i8)
declare i1 @llvm.coro.alloc(token)
