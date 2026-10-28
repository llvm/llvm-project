; The AIR math functions become the LLVM intrinsics the AIR target made them
; from (AIRIntrinsicMappings.td, inverted): precise functions keep their
; flags, fast_ ones take afn, signed and unsigned integer functions become
; their intrinsics and llvm.abs gets back its is_int_min_poison operand.
; RUN: llc -mtriple=air -air-portable %s -o %t.air.ll
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %t.air.ll | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %t.air.ll -o /dev/null

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: %a = call float @llvm.sqrt.f32(float %x)
; CHECK: %b = call afn float @llvm.sin.f32(float %a)
; CHECK: %c = call float @llvm.fma.f32(float %a, float %b, float %x)
; CHECK: %d = call nnan float @llvm.minnum.f32(float %c, float %x)
; CHECK: %e = call <2 x half> @llvm.exp2.v2f16(<2 x half>
; CHECK: %m = call i32 @llvm.smin.i32(
; CHECK: %n = call i32 @llvm.umax.i32(i32 %m, i32 7)
; CHECK: %o = call i32 @llvm.abs.i32(i32 %n, i1 false)
; CHECK: %p = call i32 @llvm.ctpop.i32(i32 %o)
; CHECK: %q = call i32 @llvm.ctlz.i32(i32 %p, i1 false)
; CHECK: %r = call i32 @llvm.bitreverse.i32(i32 %q)
; CHECK-NOT: @air.

declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()

define void @k(ptr addrspace(1) %f, ptr addrspace(1) %h, ptr addrspace(1) %u) "air-kernel" {
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %i = extractvalue { i32, i32, i32 } %t, 0
  %x = load float, ptr addrspace(1) %f
  %a = call float @llvm.sqrt.f32(float %x)
  %b = call afn float @llvm.sin.f32(float %a)
  %c = call float @llvm.fma.f32(float %a, float %b, float %x)
  %d = call nnan float @llvm.minnum.f32(float %c, float %x)
  store float %d, ptr addrspace(1) %f
  %hv = load <2 x half>, ptr addrspace(1) %h
  %e = call <2 x half> @llvm.exp2.v2f16(<2 x half> %hv)
  store <2 x half> %e, ptr addrspace(1) %h
  %v = load i32, ptr addrspace(1) %u
  %m = call i32 @llvm.smin.i32(i32 %v, i32 %i)
  %n = call i32 @llvm.umax.i32(i32 %m, i32 7)
  %o = call i32 @llvm.abs.i32(i32 %n, i1 false)
  %p = call i32 @llvm.ctpop.i32(i32 %o)
  %q = call i32 @llvm.ctlz.i32(i32 %p, i1 false)
  %r = call i32 @llvm.bitreverse.i32(i32 %q)
  store i32 %r, ptr addrspace(1) %u
  ret void
}
