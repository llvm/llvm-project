; In AIR a kernel takes a scalar as a reference to constant memory. The
; portable passes give each by-value integer, float or vector argument a
; constant buffer argument of its own, loaded once at the kernel's entry; an
; i1 is held in an i8. Pointer arguments and the arguments of functions that
; are not kernels keep their form. !air.kernel describes each buffer with the
; type it holds.
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s

; CHECK-LABEL: define internal float @helper(float %x)

; CHECK-LABEL: define void @kernel(
; CHECK-SAME: ptr addrspace(2) noalias noundef nonnull readonly align 4 dereferenceable(4) %alpha,
; CHECK-SAME: ptr addrspace(1) %out,
; CHECK-SAME: ptr addrspace(2) noalias noundef nonnull readonly align 16 dereferenceable(16) %v,
; CHECK-SAME: ptr addrspace(2) noalias noundef nonnull readonly align 1 dereferenceable(1) %flag,
; CHECK-SAME: ptr addrspace(2) noalias noundef nonnull readonly align 8 dereferenceable(8) %big,
; CHECK-SAME: <3 x i32> %tid)
; CHECK-DAG: %alpha.val = load float, ptr addrspace(2) %alpha, align 4
; CHECK-DAG: %v.val = load <4 x float>, ptr addrspace(2) %v, align 16
; CHECK-DAG: %flag.val = load i8, ptr addrspace(2) %flag, align 1
; CHECK-DAG: trunc i8 %flag.val to i1
; CHECK-DAG: %big.val = load i64, ptr addrspace(2) %big, align 8

; CHECK: !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"alpha"}
; CHECK: !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, {{.*}}!"air.arg_name", !"out"}
; CHECK: !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"v"}
; CHECK: !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"flag"}
; CHECK: !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"long", !"air.arg_name", !"big"}
; CHECK: !{i32 5, !"air.thread_position_in_grid", {{.*}}!"air.arg_name", !"tid"}


declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()

define internal float @helper(float %x) {
  %y = fmul float %x, %x
  ret float %y
}

define void @kernel(float %alpha, ptr addrspace(1) %out, <4 x float> %v, i1 %flag, i64 %big) "air-kernel" {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %tx = extractvalue { i32, i32, i32 } %t, 0
  %e = extractelement <4 x float> %v, i32 1
  %a = call float @helper(float %alpha)
  %m = fmul float %a, %e
  %bf = uitofp i64 %big to float
  %s = fadd float %m, %bf
  %r = select i1 %flag, float %s, float 0.0
  %p = getelementptr float, ptr addrspace(1) %out, i32 %tx
  store float %r, ptr addrspace(1) %p
  ret void
}
