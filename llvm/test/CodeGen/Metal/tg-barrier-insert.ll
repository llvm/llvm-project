; Insert air.wg.barrier calls before threadgroup stores.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK: call void @air.wg.barrier(i32 2, i32 1)
; CHECK: store i32 {{.*}}, ptr addrspace(3)

@tg = internal addrspace(3) global [8 x i32] zeroinitializer

define void @kernel(i32 %v) {
entry:
  %p = getelementptr [8 x i32], ptr addrspace(3) @tg, i32 0, i32 0
  store i32 %v, ptr addrspace(3) %p
  ret void
}
