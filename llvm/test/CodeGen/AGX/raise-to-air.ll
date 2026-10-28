; RUN: opt -mtriple=agx -passes=agx-raise-to-air -S < %s | FileCheck %s --check-prefix=AIR
; RUN: opt -mtriple=agx -passes=agx-raise-to-air -S < %s | FileCheck %s --check-prefix=RZ
; RUN: opt -mtriple=agx -passes=agx-raise-to-air < %s | llc -mtriple=air -air-portable | llc -mtriple=agx -mcpu=agx-g14c | FileCheck %s --check-prefix=AGX
; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s --check-prefix=AGX

; agx-raise-to-air writes AGX kernels in AIR. What AIR has (constant buffers,
; system values, SIMD-group functions) as AIR, the rest as hints, double
; precision as i64 with the soft-f64 library; generic pointers in the hint
; address space. agx-lower-air restores the AGX operations, so both paths
; give the same code.

; AIR: target triple = "air"
; AIR: define internal i32 @__air_hint.call_llvm.agx.get.fpmode____mode(i32 %0) #[[F:[0-9]+]]
; AIR-NEXT: ret i32 %0
; AIR-LABEL: define void @k(ptr addrspace(2) byref(i64) %pp, ptr addrspace(2) byref(double) %pd)
; AIR: inttoptr i64 %p to ptr addrspace(4)
; AIR: load i64, ptr addrspace(2) %pd
; AIR: call <3 x i32> @air.thread_position_in_grid.v3i32()
; AIR: call float @air.simd_shuffle_xor.f32(float %v, i16 1)
; AIR: call i64 @__air_hint.inst_fadd_ddd(i64 %d, i64 4609434218613702656)
; AIR: call i32 @__air_hint.call_llvm.agx.get.fpmode____mode(i32 5)
; AIR: define internal i64 @__air_hint.inst_fadd_ddd(i64 %0, i64 %1) #[[H:[0-9]+]]
; AIR-NEXT: call i64 @__air_f64_add(i64 %0, i64 %1)
; AIR: define internal i64 @__air_f64_add
; AIR-DAG: attributes #[[F]] = { noinline "air-hint"="call:llvm.agx.get.fpmode|-|+mode" }
; AIR-DAG: attributes #[[H]] = { noinline "air-hint"="inst:fadd|ddd" }

; AGX-LABEL: k:
; AGX: get_sr r{{[0-9]+}}, sr80 (thread_position_in_grid.x)
; AGX: simd_shuffle_xor
; AGX: stop

; The FP mode as a value (the mode AGX starts with, then what set.fpmode
; stores) at each constrained intrinsic, a trailing operand of its hint that
; only the emulation reads: rounding toward zero (field 0 of the mode for
; float) in soft-f64, rounded again to float in the same direction.
; RZ: define internal float @__air_hint.call_llvm.experimental.constrained.fadd.f32_____round.dynamic_fpexcept.ignore__mode(float %0, float %1, i32 %2)
; RZ: %[[RM:[0-9]+]] = and i32 %2, 3
; The flush bit of float (bit 4), for the operands and the result.
; RZ: and i32 %2, 16
; RZ: call i64 @__air_f64_add_r(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i32 %[[RM]])
; RZ: call i32 @__air_f64_to_f32_r(i64 %{{[0-9]+}}, i32 %[[RM]])
; RZ: fadd float %{{[0-9]+}}, %{{[0-9]+}}
; RZ: icmp eq i32 %[[RM]], 1
; The exact minimum, on the bits.
; RZ: define internal half @__air_hint.call_llvm.minimum.f16____(half %0, half %1)
; RZ-NOT: @llvm.minimum
; RZ: ret half
; RZ-LABEL: define void @rz(
; RZ: call void @__air_hint.call_llvm.agx.set.fpmode___(i32 4)
; RZ: call float @__air_hint.call_llvm.experimental.constrained.fadd.f32_____round.dynamic_fpexcept.ignore__mode(float %{{[0-9]+}}, float %{{[0-9]+}}, i32 4)
; RZ: call half @__air_hint.call_llvm.minimum.f16____(half %{{[0-9]+}}, half %{{[0-9]+}})

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
  ret void
}

declare void @llvm.agx.set.fpmode(i32)
declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare half @llvm.minimum.f16(half, half)
