; RUN: llc -mtriple=armv7-unknown-linux-gnueabihf -stop-before=finalize-isel < %s | FileCheck %s

declare void @safepoint_poll()

define void @statepoint_unused_result() gc "statepoint-example" {
; CHECK-LABEL: name: statepoint_unused_result
; CHECK: STATEPOINT 0, 0, 0, @safepoint_poll
entry:
  call token (i64, i32, ptr, i32, i32, ...)
    @llvm.experimental.gc.statepoint.p0(i64 0, i32 0,
      ptr elementtype(void ()) @safepoint_poll, i32 0, i32 0, i32 0, i32 0)
  ret void
}
