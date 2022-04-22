; Make sure that all library helper coro intrinsics are lowered.
; RUN: opt < %s -passes='default<O0>' -S | FileCheck %s

; CHECK-LABEL: @uses_library_support_coro_intrinsics(
; CHECK-NOT:     @llvm.coro
; CHECK:         ret void
define void @uses_library_support_coro_intrinsics(i8* %hdl) {
entry:
  call void @llvm.coro.resume(i8* %hdl)
  call void @llvm.coro.destroy(i8* %hdl)
  call i1 @llvm.coro.done(i8* %hdl)
  ret void
}

declare void @llvm.coro.resume(i8*)
declare void @llvm.coro.destroy(i8*)
declare i1 @llvm.coro.done(i8*)

