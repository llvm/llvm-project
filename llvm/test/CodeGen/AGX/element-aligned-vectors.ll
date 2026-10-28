; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-element-aligned-vectors -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

; With -agx-element-aligned-vectors, consecutive 16-bit and 8-bit device loads
; at only their element's alignment are one access of the i16 / i8 format
; (not measured on hardware; the default merges 4-byte aligned ones only).

; CHECK-LABEL: halves:
; CHECK:       device_load 0, i16, xyzw, r{{[0-9]+}}l_r{{[0-9]+}}h_r{{[0-9]+}}l_r{{[0-9]+}}h,
; CHECK-NOT:   device_load
; DEFAULT-LABEL: halves:
; DEFAULT-COUNT-4: device_load {{[01]}}, i16, x,
define agx_kernel void @halves(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %p1 = getelementptr inbounds i8, ptr addrspace(1) %in, i64 2
  %p2 = getelementptr inbounds i8, ptr addrspace(1) %in, i64 4
  %p3 = getelementptr inbounds i8, ptr addrspace(1) %in, i64 6
  %a = load half, ptr addrspace(1) %in, align 2
  %b = load half, ptr addrspace(1) %p1, align 2
  %c = load half, ptr addrspace(1) %p2, align 2
  %d = load half, ptr addrspace(1) %p3, align 2
  %ab = fadd half %a, %b
  %cd = fadd half %c, %d
  %s = fadd half %ab, %cd
  store half %s, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: bytes:
; CHECK:       device_load 0, i8, xyzw,
; CHECK-NOT:   device_load
define agx_kernel void @bytes(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %p1 = getelementptr inbounds i8, ptr addrspace(1) %in, i64 1
  %p2 = getelementptr inbounds i8, ptr addrspace(1) %in, i64 2
  %p3 = getelementptr inbounds i8, ptr addrspace(1) %in, i64 3
  %a = load i8, ptr addrspace(1) %in, align 1
  %b = load i8, ptr addrspace(1) %p1, align 1
  %c = load i8, ptr addrspace(1) %p2, align 1
  %d = load i8, ptr addrspace(1) %p3, align 1
  %ab = add i8 %a, %b
  %cd = xor i8 %c, %d
  %s = add i8 %ab, %cd
  store i8 %s, ptr addrspace(1) %out
  ret void
}
