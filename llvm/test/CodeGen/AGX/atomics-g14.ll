; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,G14
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,G13

; G14 has a float32 device atomic add: the sub opcode with bit 45 set
; (printed as the coherent operand 1). Elsewhere, and on threadgroup memory,
; it is a compare-and-swap loop.
; CHECK-LABEL: fadd_device:
; G14:         atomic sub, 0, r{{[0-9]+}}, u0_u1, 0, signed, r{{[0-9]+}}, 1
; G14-NOT:     cmpxchg
; G13:         atomic cmpxchg
define agx_kernel void @fadd_device(ptr addrspace(1) %p, ptr addrspace(1) %o, float %v) {
  %old = atomicrmw fadd ptr addrspace(1) %p, float %v monotonic
  store float %old, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: fadd_threadgroup:
; CHECK:       threadgroup_atomic cmpxchg
@tg = internal addrspace(3) global float undef
define agx_kernel void @fadd_threadgroup(ptr addrspace(1) %o, float %v) {
  %old = atomicrmw fadd ptr addrspace(3) @tg, float %v monotonic
  store float %old, ptr addrspace(1) %o
  ret void
}

; Integer sub is an add of the negated value on every chip, so the coherency
; bit of multi-die parts can never turn it into the float add.
; CHECK-LABEL: sub_device:
; CHECK:       isub
; CHECK:       atomic add,
; CHECK-NOT:   atomic sub
define agx_kernel void @sub_device(ptr addrspace(1) %p, ptr addrspace(1) %o, i32 %v) {
  %old = atomicrmw sub ptr addrspace(1) %p, i32 %v monotonic
  store i32 %old, ptr addrspace(1) %o
  ret void
}
