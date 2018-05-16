; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple=aarch64-linux-gnu < %s | FileCheck %s

; CHECK: OUTLINED_FUNCTION_0:
; CHECK:      orr     w0, wzr, #0x1
; CHECK-NEXT: orr     w1, wzr, #0x2
; CHECK-NEXT: orr     w2, wzr, #0x3
; CHECK-NEXT: orr     w3, wzr, #0x4
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
