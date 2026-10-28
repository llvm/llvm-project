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

; Naturally aligned device loads of 4 and 2 bytes (PTX ld.v4.u8 / ld.v2.u8) are
; one access of the byte format into consecutive 16-bit halves; of a vector
; whose only used lane is one byte, that byte alone is loaded.
; CHECK-LABEL: bytes:
; CHECK: device_load {{[01]}}, i8, xyzw, [[A:r[0-9]+]]l_[[A]]h_[[B:r[0-9]+]]l_[[B]]h,
; CHECK: device_load {{[01]}}, i8, x, r{{[0-9]+}}, u2_u3, 1,
; CHECK-NOT: device_load
; CHECK: stop
define agx_kernel void @bytes(ptr addrspace(1) %p, ptr addrspace(1) %q, ptr addrspace(1) %o) {
  %v = load <4 x i8>, ptr addrspace(1) %p, align 4
  %a = extractelement <4 x i8> %v, i32 0
  %b = extractelement <4 x i8> %v, i32 3
  %za = zext i8 %a to i32
  %zb = zext i8 %b to i32
  %s = add i32 %za, %zb
  %w = load <2 x i8>, ptr addrspace(1) %q, align 2
  %c = extractelement <2 x i8> %w, i32 1
  %zc = zext i8 %c to i32
  %t = add i32 %s, %zc
  store i32 %t, ptr addrspace(1) %o
  ret void
}

; Less aligned, they stay byte loads.
; CHECK-LABEL: bytes_unaligned:
; CHECK-NOT: xyzw
; CHECK: stop
define agx_kernel void @bytes_unaligned(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %v = load <4 x i8>, ptr addrspace(1) %p, align 1
  %a = extractelement <4 x i8> %v, i32 0
  %b = extractelement <4 x i8> %v, i32 3
  %za = zext i8 %a to i32
  %zb = zext i8 %b to i32
  %s = add i32 %za, %zb
  store i32 %s, ptr addrspace(1) %o
  ret void
}

; and the stores of 4 and 2 bytes, from consecutive 16-bit halves
; CHECK-LABEL: byte_stores:
; CHECK: device_store 0, i8, xyzw, [[A:r[0-9]+]]l_[[A]]h_[[B:r[0-9]+]]l_[[B]]h,
; CHECK: device_store 0, i8, xy, [[C:r[0-9]+]]l_[[C]]h,
; CHECK-NOT: device_store
; CHECK: stop
define agx_kernel void @byte_stores(ptr addrspace(1) %p, ptr addrspace(1) %q, i32 %x) {
  %t = trunc i32 %x to i8
  %u = add i8 %t, 1
  %v0 = insertelement <4 x i8> poison, i8 %t, i32 0
  %v1 = insertelement <4 x i8> %v0, i8 %u, i32 1
  %v2 = insertelement <4 x i8> %v1, i8 7, i32 2
  %v3 = insertelement <4 x i8> %v2, i8 200, i32 3
  store <4 x i8> %v3, ptr addrspace(1) %p, align 4
  %w0 = insertelement <2 x i8> poison, i8 %u, i32 0
  %w1 = insertelement <2 x i8> %w0, i8 %t, i32 1
  store <2 x i8> %w1, ptr addrspace(1) %q, align 2
  ret void
}
