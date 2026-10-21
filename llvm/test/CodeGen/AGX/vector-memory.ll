; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+agx-strict -verify-machineinstrs < %s | FileCheck %s

; 2 to 4 consecutive 32-bit values at a 4-byte aligned device or threadgroup
; address are one xy / xyz / xyzw access (ptx-re-blockers B33), whether the
; source has a vector or neighbouring scalars.

declare i32 @llvm.agx.thread.position.in.grid.x()
@tg = internal addrspace(3) global [64 x i32] undef, align 4

; CHECK-LABEL: copy_v4:
; CHECK:      device_load 0, i32, xyzw, [[V:r[0-9]+_r[0-9]+_r[0-9]+_r[0-9]+]], u0_u1,
; CHECK:      device_store 0, i32, xyzw, [[V]], u2_u3,
define agx_kernel void @copy_v4(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %p = getelementptr <4 x float>, ptr addrspace(1) %in, i32 %i
  %v = load <4 x float>, ptr addrspace(1) %p, align 4
  %q = getelementptr <4 x float>, ptr addrspace(1) %out, i32 %i
  store <4 x float> %v, ptr addrspace(1) %q, align 4
  ret void
}

; CHECK-LABEL: three_scalars:
; CHECK:      device_load 0, i32, xyz, r{{[0-9]+}}_r{{[0-9]+}}_r{{[0-9]+}}, u0_u1,
; CHECK-NOT:  device_load
; CHECK:      stop
define agx_kernel void @three_scalars(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %a = load i32, ptr addrspace(1) %in, align 4
  %p1 = getelementptr i32, ptr addrspace(1) %in, i32 1
  %b = load i32, ptr addrspace(1) %p1, align 4
  %p2 = getelementptr i32, ptr addrspace(1) %in, i32 2
  %c = load i32, ptr addrspace(1) %p2, align 4
  %s = add i32 %a, %b
  %t = add i32 %s, %c
  store i32 %t, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: threadgroup_pair:
; CHECK:      threadgroup_store i32, xy,
; CHECK:      threadgroup_load i32, xy,
define agx_kernel void @threadgroup_pair(ptr addrspace(1) %out, i32 %x, i32 %y) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %e = shl i32 %i, 1
  %p0 = getelementptr i32, ptr addrspace(3) @tg, i32 %e
  %p1 = getelementptr i32, ptr addrspace(3) %p0, i32 1
  store i32 %x, ptr addrspace(3) %p0, align 4
  store i32 %y, ptr addrspace(3) %p1, align 4
  call void @llvm.agx.threadgroup.barrier()
  %a = load i32, ptr addrspace(3) %p0, align 4
  %b = load i32, ptr addrspace(3) %p1, align 4
  %s = mul i32 %a, %b
  %q = getelementptr i32, ptr addrspace(1) %out, i32 %i
  store i32 %s, ptr addrspace(1) %q, align 4
  ret void
}
declare void @llvm.agx.threadgroup.barrier()

; Two bytes apart is not 4-byte aligned: separate accesses.
; CHECK-LABEL: unaligned:
; CHECK-NOT:  xy
; CHECK:      stop
define agx_kernel void @unaligned(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %v = load <2 x i32>, ptr addrspace(1) %in, align 2
  store <2 x i32> %v, ptr addrspace(1) %out, align 2
  ret void
}
