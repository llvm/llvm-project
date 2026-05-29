; Regression test for the pipeline ordering invariant: scalar-buffer-packing
; runs BEFORE air-system-values, so the !air.kernel metadata records the
; post-packing parameter list (device buffer + packed scalar buffer + the
; injected <3 x i32> tid argument). The order must therefore be:
;   %p (kept device buf), %_scalar_buf (packed scalars), %tid (system value)
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-SAME: ptr addrspace(1){{.*}} %p, ptr addrspace(1){{.*}} %_scalar_buf, <3 x i32> %tid

declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()

define void @kernel(ptr addrspace(1) %p, i32 %off) {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %tx = extractvalue { i32, i32, i32 } %t, 0
  %sum = add i32 %tx, %off
  store i32 %sum, ptr addrspace(1) %p, align 4
  ret void
}
