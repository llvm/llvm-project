; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-narrow-vector-loads=0 \
; RUN:   -agx-split-permuted-stores=0 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=OFF

; A vector store of loaded lanes in another order would need register copies:
; it is stored as the runs that need none.

; CHECK-LABEL: swap:
; CHECK:      device_load 0, i32, xy, r[[A:[0-9]+]]_r[[B:[0-9]+]], u0_u1, 0
; CHECK-NOT:  bitop
; CHECK-NOT:  mov
; CHECK:      device_store 0, i32, x, r[[B]], u2_u3, 0
; CHECK-NEXT: device_store 0, i32, x, r[[A]], u2_u3, 1
; OFF-LABEL:  swap:
; OFF:        device_store 0, i32, xy,
define agx_kernel void @swap(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %a = load i32, ptr addrspace(1) %in, align 4
  %p1 = getelementptr inbounds i8, ptr addrspace(1) %in, i64 4
  %b = load i32, ptr addrspace(1) %p1, align 4
  store i32 %b, ptr addrspace(1) %out, align 4
  %q1 = getelementptr inbounds i8, ptr addrspace(1) %out, i64 4
  store i32 %a, ptr addrspace(1) %q1, align 4
  ret void
}

; Lanes in load order stay one vector store.
; CHECK-LABEL: copy:
; CHECK:      device_load 0, i32, xy, [[V:r[0-9]+_r[0-9]+]], u0_u1, 0
; CHECK:      device_store 0, i32, xy, [[V]], u2_u3, 0
define agx_kernel void @copy(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %a = load i32, ptr addrspace(1) %in, align 4
  %p1 = getelementptr inbounds i8, ptr addrspace(1) %in, i64 4
  %b = load i32, ptr addrspace(1) %p1, align 4
  store i32 %a, ptr addrspace(1) %out, align 4
  %q1 = getelementptr inbounds i8, ptr addrspace(1) %out, i64 4
  store i32 %b, ptr addrspace(1) %q1, align 4
  ret void
}

; Only lane 2 of the v4 load is read: it loads that lane, which then merges
; with the neighbouring loads (PTX `ld.v4.u32 {_, _, z, _}`).
; CHECK-LABEL: partial:
; CHECK:      device_load 0, i32, xyz, r{{[0-9]+}}_r{{[0-9]+}}_r{{[0-9]+}}, u0_u1, 0
; CHECK-NOT:  device_load
; OFF-LABEL:  partial:
; OFF:        device_load 1, i32, xyzw,
define agx_kernel void @partial(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %ab = load <2 x i32>, ptr addrspace(1) %in, align 8
  %a = extractelement <2 x i32> %ab, i64 0
  %b = extractelement <2 x i32> %ab, i64 1
  %v = load <4 x i32>, ptr addrspace(1) %in, align 16
  %z = extractelement <4 x i32> %v, i64 2
  store i32 %b, ptr addrspace(1) %out, align 4
  %q1 = getelementptr inbounds i8, ptr addrspace(1) %out, i64 4
  store i32 %a, ptr addrspace(1) %q1, align 4
  %q2 = getelementptr inbounds i8, ptr addrspace(1) %out, i64 8
  store i32 %z, ptr addrspace(1) %q2, align 4
  ret void
}
