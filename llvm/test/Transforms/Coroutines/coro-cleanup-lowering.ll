; Make sure that all library helper coro intrinsics are lowered.
; RUN: opt < %s -passes=coro-cleanup -S | FileCheck %s

; CHECK-LABEL: @uses_library_support_coro_intrinsics(
; CHECK-NOT:     @llvm.coro
; CHECK:         ret void

define void @uses_library_support_coro_intrinsics(i8* %hdl) {
entry:
  %0 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 0)
  %1 = bitcast i8* %0 to void (i8*)*
  call fastcc void %1(i8* %hdl)
  %2 = call i8* @llvm.coro.subfn.addr(i8* %hdl, i8 1)
  %3 = bitcast i8* %2 to void (i8*)*
  call fastcc void %3(i8* %hdl)
  %4 = bitcast i8* %hdl to i8**
  %5 = load i8*, i8** %4
  %6 = icmp eq i8* %5, null
  ret void
}
declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)
; Function Attrs: argmemonly nounwind
declare i1 @llvm.coro.done(i8* nocapture readonly) #0
; Function Attrs: argmemonly nounwind readonly
declare i8* @llvm.coro.subfn.addr(i8* nocapture readonly, i8) #1

attributes #0 = { argmemonly nounwind }
attributes #1 = { argmemonly nounwind readonly }
