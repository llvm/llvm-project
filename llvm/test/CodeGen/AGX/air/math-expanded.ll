; AIR's math functions without an LLVM intrinsic (asinh, acosh, atanh, sinpi,
; cospi, tanpi, and their fast_ forms) are built from the functions AGX has.
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %s -o /dev/null

; CHECK-LABEL: define agx_kernel void @k(
; CHECK-NOT: call {{.*}}@air.
; CHECK: call float @llvm.log.f32
; CHECK: call float @llvm.roundeven.f32
; CHECK: call float @llvm.sin.f32
; CHECK: ret void

declare float @air.asinh.f32(float)
declare float @air.fast_acosh.f32(float)
declare float @air.atanh.f32(float)
declare float @air.sinpi.f32(float)
declare float @air.cospi.f32(float)
declare <2 x half> @air.tanpi.v2f16(<2 x half>)

define void @k(ptr addrspace(1) %p, ptr addrspace(1) %h) {
  %x = load float, ptr addrspace(1) %p
  %a = call float @air.asinh.f32(float %x)
  %b = call float @air.fast_acosh.f32(float %a)
  %c = call float @air.atanh.f32(float %b)
  %d = call float @air.sinpi.f32(float %c)
  %e = call float @air.cospi.f32(float %d)
  store float %e, ptr addrspace(1) %p
  %y = load <2 x half>, ptr addrspace(1) %h
  %t = call <2 x half> @air.tanpi.v2f16(<2 x half> %y)
  store <2 x half> %t, ptr addrspace(1) %h
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1, !2}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1}
