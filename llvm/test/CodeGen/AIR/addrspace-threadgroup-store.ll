; Address-space 3 (threadgroup) stores survive the pipeline. The
; tg-barrier-insert pass wraps the threadgroup access with air.wg.barrier
; calls so that cross-thread visibility is well-defined.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK: call void @air.wg.barrier
; CHECK: store i32 {{.*}}, ptr addrspace(3)
; CHECK: ret void

define void @kernel(i32 %v, ptr addrspace(3) %tg) {
entry:
  store i32 %v, ptr addrspace(3) %tg, align 4
  ret void
}
