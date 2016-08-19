; Tests that the coro.destroy and coro.resume are devirtualized where possible,
; SCC pipeline restarts and inlines the direct calls.
; RUN: opt < %s -S -inline -coro-elide -dce | FileCheck %s

declare void @print(i32) nounwind

; resume part of the coroutine
define fastcc void @f.resume(i8*) {
  tail call void @print(i32 0)
  ret void
}

; destroy part of the coroutine
define fastcc void @f.destroy(i8*) {
  tail call void @print(i32 1)
  ret void
}

@f.resumers = internal constant [2 x void (i8*)*] [void (i8*)* @f.resume,
                                                   void (i8*)* @f.destroy]

; a coroutine start function
define i8* @f() {
entry:
  %id = call token @llvm.coro.id(i32 0, i8* null,
                          i8* bitcast ([2 x void (i8*)*]* @f.resumers to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  ret i8* %hdl
}

; CHECK-LABEL: @callResume(
define void @callResume() {
entry:
; CHECK: call i8* @llvm.coro.begin
  %hdl = call i8* @f()

; CHECK-NEXT: call void @print(i32 0)
  %0 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %1 = bitcast i8* %0 to void (i8*)*
  call fastcc void %1(i8* %hdl)

; CHECK-NEXT: call void @print(i32 1)
  %2 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %3 = bitcast i8* %2 to void (i8*)*
  call fastcc void %3(i8* %hdl)

; CHECK-NEXT: ret void
  ret void
}

; CHECK-LABEL: @eh(
define void @eh() personality i8* null {
entry:
; CHECK: call i8* @llvm.coro.begin
  %hdl = call i8* @f()

; CHECK-NEXT: call void @print(i32 0)
  %0 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %1 = bitcast i8* %0 to void (i8*)*
  invoke void %1(i8* %hdl)
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
  %id = call token @llvm.coro.id(i32 0, i8* null, i8* null)
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)

; CHECK: call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %0 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %1 = bitcast i8* %0 to void (i8*)*
  call fastcc void %1(i8* %hdl)

; CHECK: call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %2 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %3 = bitcast i8* %2 to void (i8*)*
  call fastcc void %3(i8* %hdl)

; CHECK: ret void
  ret void
}

; CHECK-LABEL: @no_devirt_no_begin(
; no devirtualization here, since coro.begin is not visible
define void @no_devirt_no_begin(i8* %hdl) {
entry:

; CHECK: call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %0 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %1 = bitcast i8* %0 to void (i8*)*
  call fastcc void %1(i8* %hdl)

; CHECK: call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %2 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %3 = bitcast i8* %2 to void (i8*)*
  call fastcc void %3(i8* %hdl)

; CHECK: ret void
  ret void
}

declare token @llvm.coro.id(i32, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i8* @llvm.coro.frame()
declare i8* @llvm.coro.subfn.addr(i8*, i8)
