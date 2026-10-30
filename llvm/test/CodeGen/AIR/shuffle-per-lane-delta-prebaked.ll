; With a pre-baked !air.kernel, the lane comes from the kernel's
; thread_index_in_simdgroup parameter; a kernel without one gets it, described
; in its metadata.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

declare i32 @air.simd_shuffle_xor.u.i32(i32, i16)

; CHECK-LABEL: define void @has_lane(
; CHECK-SAME: i32 %lane)
; CHECK: xor i32 %lane,
; CHECK: call i32 @air.simd_shuffle.u.i32(
define void @has_lane(ptr addrspace(1) %p, i32 %lane) "air-kernel" {
  %m32 = load i32, ptr addrspace(1) %p
  %m = trunc i32 %m32 to i16
  %r = call i32 @air.simd_shuffle_xor.u.i32(i32 %m32, i16 %m)
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: define void @no_lane(
; CHECK-SAME: ptr addrspace(1) {{.*}}%p, i32 %simdlane)
; CHECK: xor i32 %simdlane,
; CHECK: call i32 @air.simd_shuffle.u.i32(
define void @no_lane(ptr addrspace(1) %p) "air-kernel" {
  %m32 = load i32, ptr addrspace(1) %p
  %m = trunc i32 %m32 to i16
  %r = call i32 @air.simd_shuffle_xor.u.i32(i32 %m32, i16 %m)
  store i32 %r, ptr addrspace(1) %p
  ret void
}

; CHECK: !{ptr @no_lane, !{{[0-9]+}}, [[ARGS:![0-9]+]]}
; CHECK-DAG: [[ARGS]] = !{!{{[0-9]+}}, [[LANE:![0-9]+]]}
; CHECK-DAG: [[LANE]] = !{i32 1, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simdlane"}

!air.kernel = !{!0, !5}
!0 = !{ptr @has_lane, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"p"}
!4 = !{i32 1, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}
!5 = !{ptr @no_lane, !1, !6}
!6 = !{!3}
