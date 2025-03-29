; Tests that CoroEarly pass correctly lowers coro.resume, coro.destroy
; RUN: opt < %s -S -passes=coro-early | FileCheck %s

; CHECK-LABEL: @callResume(
define void @callResume(ptr %hdl) {
; CHECK-NEXT: entry
entry:
; CHECK-NEXT: %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
; CHECK-NEXT: call fastcc void %0(ptr %hdl)
  call void @llvm.coro.resume(ptr %hdl)

; CHECK-NEXT: %1 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
; CHECK-NEXT: call fastcc void %1(ptr %hdl)
  call void @llvm.coro.destroy(ptr %hdl)

  ret void
; CHECK-NEXT: ret void
}

; CHECK-LABEL: @eh(
define void @eh(ptr %hdl) personality ptr null {
; CHECK-NEXT: entry
entry:
;  CHECK-NEXT: %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
;  CHECK-NEXT: invoke fastcc void %0(ptr %hdl)
  invoke void @llvm.coro.resume(ptr %hdl)
          to label %cont unwind label %ehcleanup
cont:
  ret void

ehcleanup:
  %0 = cleanuppad within none []
  cleanupret from %0 unwind to caller
}


declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)
