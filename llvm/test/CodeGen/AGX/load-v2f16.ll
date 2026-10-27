; RUN: llc -mtriple=agx -mcpu=agx-g14g -verify-machineinstrs < %s | FileCheck %s

; A 4-byte aligned pair of 16-bit elements is one 32-bit access whose halves
; are the elements, also when they are used one at a time (here widened to
; f32); four elements, 8-byte aligned, are one xy access. Less aligned pairs
; stay two 16-bit accesses.

; CHECK-LABEL: pair:
; CHECK: device_load 0, i32, x, r[[R:[0-9]+]], u0_u1, 0, signed
; CHECK-NOT: device_load
; CHECK-DAG: fadd32 r{{[0-9]+}}, r[[R]]l, -0.0
; CHECK-DAG: fadd32 r{{[0-9]+}}, r[[R]]h, -0.0
define void @pair(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %v = load <2 x half>, ptr addrspace(1) %p, align 4
  %e = fpext <2 x half> %v to <2 x float>
  store <2 x float> %e, ptr addrspace(1) %q, align 8
  ret void
}

; CHECK-LABEL: quad:
; CHECK: device_load 0, i32, xy, r{{[0-9]+}}_r{{[0-9]+}}, u0_u1, 0, signed
; CHECK-NOT: device_load
define void @quad(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %v = load <4 x i16>, ptr addrspace(1) %p, align 8
  %e = zext <4 x i16> %v to <4 x i32>
  store <4 x i32> %e, ptr addrspace(1) %q, align 16
  ret void
}

; CHECK-LABEL: unaligned:
; CHECK: device_load {{[01]}}, i16, x,
; CHECK: device_load {{[01]}}, i16, x,
define void @unaligned(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %v = load <2 x half>, ptr addrspace(1) %p, align 2
  %e = fpext <2 x half> %v to <2 x float>
  store <2 x float> %e, ptr addrspace(1) %q, align 8
  ret void
}
