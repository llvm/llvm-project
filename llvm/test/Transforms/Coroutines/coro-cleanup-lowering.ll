; Make sure that all library helper coro intrinsics are lowered.
; RUN: opt < %s -passes=coro-cleanup -S | FileCheck %s

; CHECK-LABEL: @uses_library_support_coro_intrinsics(
; CHECK-NOT:     @llvm.coro
; CHECK:         ret void

define void @uses_library_support_coro_intrinsics(ptr %hdl) {
entry:
  %0 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 0)
  call fastcc void %0(ptr %hdl)
  %1 = call ptr @llvm.coro.subfn.addr(ptr %hdl, i8 1)
  call fastcc void %1(ptr %hdl)
  %2 = load ptr, ptr %hdl
  %3 = icmp eq ptr %2, null
  ret void
}
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)
; Function Attrs: argmemonly nounwind
declare i1 @llvm.coro.done(ptr nocapture readonly) #0
; Function Attrs: argmemonly nounwind readonly
declare ptr @llvm.coro.subfn.addr(ptr nocapture readonly, i8) #1

attributes #0 = { argmemonly nounwind }
attributes #1 = { argmemonly nounwind readonly }
