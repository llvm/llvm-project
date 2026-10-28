; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; A field of a value, (x >> s) & (2^k - 1) or (x & ((2^k - 1) << s)) >> s,
; is one bfeil with offset s and a k-bit mask, not a shift and an and.

; CHECK-LABEL: field:
; CHECK:       bfeil [[R:r[0-9]+]], 0, u2, 7, mask 0x3F
; CHECK-NOT:   and
; CHECK:       device_store 0, i32, x, [[R]],
define agx_kernel void @field(ptr addrspace(1) %p, i32 %x) {
  %s = lshr i32 %x, 7
  %f = and i32 %s, 63
  store i32 %f, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: masked_then_shifted:
; CHECK:       bfeil [[R:r[0-9]+]], 0, u2, 4, mask 0xF
; CHECK-NOT:   and
; CHECK:       device_store 0, i32, x, [[R]],
define agx_kernel void @masked_then_shifted(ptr addrspace(1) %p, i32 %x) {
  %m = and i32 %x, 240
  %f = lshr i32 %m, 4
  store i32 %f, ptr addrspace(1) %p
  ret void
}

; A shift other than the mask's offset is not a field.
; CHECK-LABEL: not_a_field:
; CHECK:       and
; CHECK:       bfeil
define agx_kernel void @not_a_field(ptr addrspace(1) %p, i32 %x) {
  %m = and i32 %x, 240
  %f = lshr i32 %m, 2
  store i32 %f, ptr addrspace(1) %p
  ret void
}
