; Tests that CoroEarly pass correctly lowers coro.resume and coro.destroy
; intrinsics.
; RUN: opt < %s -S -coro-early | FileCheck %s

; CHECK-LABEL: @callResume(
define void @callResume(i8* %hdl) {
; CHECK-NEXT: entry
entry:
; CHECK-NEXT: %0 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
; CHECK-NEXT: %1 = bitcast i8* %0 to void (i8*)*
; CHECK-NEXT: call fastcc void %1(i8* %hdl)
  call void @llvm.coro.resume(i8* %hdl)

; CHECK-NEXT: %2 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
; CHECK-NEXT: %3 = bitcast i8* %2 to void (i8*)*
; CHECK-NEXT: call fastcc void %3(i8* %hdl)
  call void @llvm.coro.destroy(i8* %hdl)

  ret void
; CHECK-NEXT: ret void
}

; CHECK-LABEL: @eh(
define void @eh(i8* %hdl) personality i8* null {
; CHECK-NEXT: entry
entry:
;  CHECK-NEXT: %0 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
;  CHECK-NEXT: %1 = bitcast i8* %0 to void (i8*)*
;  CHECK-NEXT: invoke fastcc void %1(i8* %hdl)
  invoke void @llvm.coro.resume(i8* %hdl)
          to label %cont unwind label %ehcleanup
cont:
  ret void

ehcleanup:
  %0 = cleanuppad within none []
  cleanupret from %0 unwind to caller
}

declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)
