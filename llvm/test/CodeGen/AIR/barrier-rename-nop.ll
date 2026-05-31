; Negative test: a kernel that never calls `air.threadgroup.barrier` is not
; touched by the rename pass; no `air.wg.barrier` materializes.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: air.wg.barrier
; CHECK-NOT: air.threadgroup.barrier
; CHECK: store i32 0, ptr %out
; CHECK: ret void

define void @kernel(ptr %out) {
entry:
  store i32 0, ptr %out
  ret void
}
