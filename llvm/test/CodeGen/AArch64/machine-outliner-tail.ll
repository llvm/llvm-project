; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple=aarch64-linux-gnu < %s | FileCheck %s

; CHECK: OUTLINED_FUNCTION_0:
; CHECK:      mov     w0, #1
; CHECK-NEXT: mov     w1, #2
; CHECK-NEXT: mov     w2, #3
; CHECK-NEXT: mov     w3, #4
; CHECK-NEXT: b       z

define void @a() {
entry:
  tail call void @z(i32 1, i32 2, i32 3, i32 4)
  ret void
}

declare void @z(i32, i32, i32, i32)

define dso_local void @b(i32* nocapture readnone %p) {
entry:
  tail call void @z(i32 1, i32 2, i32 3, i32 4)
  ret void
}
