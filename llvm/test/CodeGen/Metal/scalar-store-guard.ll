; Scalar kernels (device write, no per-thread index) get a tid.x == 0 guard.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @vector_kernel
; CHECK: store i32 {{.*}}, ptr addrspace(1)
; CHECK: ret void

; CHECK-LABEL: define void @scalar_kernel
; CHECK: <3 x i32> %tidtg
; CHECK: extractelement <3 x i32> %tidtg
; CHECK: icmp eq i32
; CHECK: br i1
; CHECK: store i32 %v, ptr addrspace(1)

declare [3 x i32] @air.thread_position_in_grid()

define void @scalar_kernel(ptr addrspace(1) %p, i32 %v) {
entry:
  store i32 %v, ptr addrspace(1) %p
  ret void
}

define void @vector_kernel(ptr addrspace(1) %p) {
entry:
  %t = call [3 x i32] @air.thread_position_in_grid()
  %tx = extractvalue [3 x i32] %t, 0
  store i32 %tx, ptr addrspace(1) %p
  ret void
}
