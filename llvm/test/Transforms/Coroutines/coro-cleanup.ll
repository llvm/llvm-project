; Make sure that all library helper coro intrinsics are lowered.
; RUN: opt < %s -passes='default<O0>' -S | FileCheck %s

; CHECK-LABEL: @uses_library_support_coro_intrinsics(
; CHECK-NOT:     @llvm.coro
; CHECK:         ret void
define void @uses_library_support_coro_intrinsics(ptr %hdl) {
entry:
  call void @llvm.coro.resume(ptr %hdl)
  call void @llvm.coro.destroy(ptr %hdl)
  call i1 @llvm.coro.done(ptr %hdl)
  ret void
}

declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)
declare i1 @llvm.coro.done(ptr)

