; RUN: opt -mtriple=agx -passes=agx-raise-to-air -S < %s | FileCheck %s --check-prefix=AIR
; RUN: opt -mtriple=agx -passes=agx-raise-to-air < %s | llc -mtriple=air -air-portable | llc -mtriple=agx -mcpu=agx-g14c | FileCheck %s --check-prefix=AGX
; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s --check-prefix=AGX

; agx-raise-to-air writes AGX kernels in AIR. What AIR has (constant buffers,
; system values, SIMD-group functions) as AIR, the rest as hints, double
; precision as i64 with the soft-f64 library; generic pointers in the hint
; address space. agx-lower-air restores the AGX operations, so both paths
; give the same code.

; AIR: target triple = "air"
; AIR: define internal i32 @__air_hint.call_llvm.agx.get.fpmode__() #[[F:[0-9]+]]
; AIR-NEXT: ret i32 0
; AIR-LABEL: define void @k(ptr addrspace(2) byref(i64) %pp, ptr addrspace(2) byref(double) %pd)
; AIR: inttoptr i64 %p to ptr addrspace(4)
; AIR: load i64, ptr addrspace(2) %pd
; AIR: call <3 x i32> @air.thread_position_in_grid.v3i32()
; AIR: call float @air.simd_shuffle_xor.f32(float %v, i16 1)
; AIR: call i64 @__air_hint.inst_fadd_ddd(i64 %d, i64 4609434218613702656)
; AIR: call i32 @__air_hint.call_llvm.agx.get.fpmode__()
; AIR: define internal i64 @__air_hint.inst_fadd_ddd(i64 %0, i64 %1) #[[H:[0-9]+]]
; AIR-NEXT: call i64 @__air_f64_add(i64 %0, i64 %1)
; AIR: define internal i64 @__air_f64_add
; AIR-DAG: attributes #[[F]] = { noinline "air-hint"="call:llvm.agx.get.fpmode|-" }
; AIR-DAG: attributes #[[H]] = { noinline "air-hint"="inst:fadd|ddd" }

; AGX-LABEL: k:
; AGX: get_sr r{{[0-9]+}}, sr80 (thread_position_in_grid.x)
; AGX: simd_shuffle_xor
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
