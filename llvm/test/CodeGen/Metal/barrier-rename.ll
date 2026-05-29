; Rename air.threadgroup.barrier calls to air.wg.barrier. Args are passed
; through unchanged — Metal 4 JIT accepts legacy (1, 4) on the new name.
; See PASS_GUARDS.md sub-track J for the real-context bisect.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: air.threadgroup.barrier
; CHECK: call void @air.wg.barrier(i32 1, i32 4)
; CHECK: ret void

declare void @air.threadgroup.barrier(i32, i32)

define void @kernel() {
entry:
  call void @air.threadgroup.barrier(i32 1, i32 4)
  ret void
}
