; RUN: opt -mtriple=agx -passes=agx-raise-to-air -S < %s | FileCheck %s --check-prefix=AIR
; RUN: opt -mtriple=agx -passes=agx-raise-to-air -agx-raise-hints=all -S < %s | FileCheck %s --check-prefix=HINTS
; RUN: opt -mtriple=agx -passes=agx-raise-to-air < %s | llc -mtriple=air -air-portable | llc -mtriple=agx -mcpu=agx-g14c | FileCheck %s --check-prefix=AGX
; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s --check-prefix=AGX

; agx-raise-to-air writes AGX kernels in portable AIR. What AIR has
; (constant buffers, system values, SIMD-group functions, minimum) as AIR,
; double precision as i64 with the soft-f64 library, and the AGX operations
; AIR lacks as their AIR emulations, inlined and folded where the FP mode is
; known. Only the hints of agx-raise-hints stay hints (by default the
; SIMD-group matrix FMA).

; AIR: target triple = "air"
; AIR-NOT: __air_hint
; AIR-LABEL: define void @k(ptr addrspace(2) byref(i64) %pp, ptr addrspace(2) byref(double) %pd)
; Nothing makes a generic pointer to other memory: it is a device pointer.
; AIR: inttoptr i64 %p to ptr addrspace(1)
; AIR: load i64, ptr addrspace(2) %pd
; AIR: call <3 x i32> @air.thread_position_in_grid.v3i32()
; AIR: call float @air.simd_shuffle_xor.f32(float %v, i16 1)
; AIR: %[[S:[0-9]+]] = call i64 @__air_f64_add(i64 %d, i64 4609434218613702656)
; The FP mode is known: get.fpmode is 5 (rounding to nearest for both formats).
; AIR: %[[M:[0-9]+]] = call i64 @__air_f64_from_u64(i64 5)
; AIR: call i64 @__air_f64_add(i64 %[[S]], i64 %[[M]])
; Rounding toward zero (set.fpmode 4: field 0, float), in soft-f64.
; AIR-LABEL: define void @rz(
; AIR: call i64 @__air_f64_add_r(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i32 0)
; AIR: call i32 @__air_f64_to_f32_r(i64 %{{[0-9]+}}, i32 0)
; AIR: call half @llvm.minimum.f16(
; AIR: call i32 @air.thread_index_in_simdgroup()
; AIR-NOT: strictfp
; AIR: define internal i64 @__air_f64_add

; HINTS: define internal i32 @__air_hint.call_llvm.agx.get.fpmode____mode(i32 %0) #[[F:[0-9]+]]
; HINTS-NEXT: ret i32 %0
; HINTS: define internal float @__air_hint.call_llvm.experimental.constrained.fadd.f32_____round.dynamic_fpexcept.ignore__mode(float %0, float %1, i32 %2)
; HINTS: call i64 @__air_f64_add_r(
; HINTS-LABEL: define void @k(
; HINTS: call i64 @__air_hint.inst_fadd_ddd(i64 %d, i64 4609434218613702656)
; HINTS: call i32 @__air_hint.call_llvm.agx.get.fpmode____mode(i32 5)
; HINTS-LABEL: define void @rz(
; HINTS: call float @__air_hint.call_llvm.experimental.constrained.fadd.f32_____round.dynamic_fpexcept.ignore__mode(float %{{[0-9]+}}, float %{{[0-9]+}}, i32 4)
; HINTS: call i32 @__air_hint.call_llvm.agx.thread.index.in.simdgroup____value(
; HINTS-DAG: attributes #[[F]] = { noinline "air-hint"="call:llvm.agx.get.fpmode|-|+mode" }

; AGX-LABEL: k:
; AGX: get_sr r{{[0-9]+}}, sr80 (thread_position_in_grid.x)
; AGX: simd_shuffle_xor
; AGX: stop
; AGX-LABEL: rz:
; AGX: stop
define agx_kernel void @k(ptr addrspace(4) byref(i64) %pp, ptr addrspace(4) byref(double) %pd) {
  %p = load i64, ptr addrspace(4) %pp
  %g = inttoptr i64 %p to ptr
  %d = load double, ptr addrspace(4) %pd
  %t = call i32 @llvm.agx.thread.position.in.grid.x()
  %gp = getelementptr float, ptr %g, i32 %t
  %v = load float, ptr %gp
  %s = call float @llvm.agx.simd.shuffle.xor.f32(float %v, i32 1)
  store float %s, ptr %gp
  %dd = fadd double %d, 1.5
  %m = call i32 @llvm.agx.get.fpmode()
  %mf = uitofp i32 %m to double
  %r = fadd double %dd, %mf
  %dp = getelementptr i8, ptr %g, i64 64
  store double %r, ptr %dp
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
declare float @llvm.agx.simd.shuffle.xor.f32(float, i32)
declare i32 @llvm.agx.get.fpmode()

define agx_kernel void @rz(ptr addrspace(4) byref(i64) %pp, float %x, float %y, half %h, half %h2) strictfp {
  %p = load i64, ptr addrspace(4) %pp
  %g = inttoptr i64 %p to ptr addrspace(1)
  call void @llvm.agx.set.fpmode(i32 4) strictfp
  %s = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  store float %s, ptr addrspace(1) %g
  %m = call half @llvm.minimum.f16(half %h, half %h2)
  %hp = getelementptr i8, ptr addrspace(1) %g, i64 8
  store half %m, ptr addrspace(1) %hp
  %lane = call i32 @llvm.agx.thread.index.in.simdgroup()
  %lp = getelementptr i8, ptr addrspace(1) %g, i64 12
  store i32 %lane, ptr addrspace(1) %lp
  ret void
}

declare void @llvm.agx.set.fpmode(i32)
declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare half @llvm.minimum.f16(half, half)
declare i32 @llvm.agx.thread.index.in.simdgroup()
