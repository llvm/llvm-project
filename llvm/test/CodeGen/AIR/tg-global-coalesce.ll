; Merge __tg_cvt_* into __tg_dot_ab_* when MMA intrinsics are present.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-NOT: @__tg_cvt_x
; CHECK: @__tg_dot_ab_y = internal addrspace(3) global [128 x float]

@__tg_cvt_x = internal addrspace(3) global [128 x float] zeroinitializer
@__tg_dot_ab_y = internal addrspace(3) global [128 x float] zeroinitializer

declare float @air.simdgroup_matrix_8x8_load(ptr addrspace(3))

define void @kernel(ptr addrspace(1) %out) {
entry:
  %p = getelementptr [128 x float], ptr addrspace(3) @__tg_cvt_x, i32 0, i32 0
  store float 1.0, ptr addrspace(3) %p
  %q = getelementptr [128 x float], ptr addrspace(3) @__tg_dot_ab_y, i32 0, i32 0
  %v = call float @air.simdgroup_matrix_8x8_load(ptr addrspace(3) %q)
  store float %v, ptr addrspace(1) %out
  ret void
}
