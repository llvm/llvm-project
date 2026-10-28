; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; A sign-extended byte converts with s8_to_f, which reads the low eight bits
; as signed, instead of a shift pair and s16_to_f / s32_to_f.

; CHECK-LABEL: s8_load_to_f32:
; CHECK:       device_load 0, i8, x, [[B:r[0-9]+l]],
; CHECK-NOT:   asr
; CHECK:       convert s8_to_f, [[F:r[0-9]+]], [[B]], rte
; CHECK:       device_store 0, i32, x, [[F]],
define agx_kernel void @s8_load_to_f32(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %b = load i8, ptr addrspace(1) %q
  %f = sitofp i8 %b to float
  store float %f, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: s8_of_i32_to_f32:
; CHECK-NOT:   asr
; CHECK:       convert s8_to_f, [[F:r[0-9]+]], u2l, rte
define agx_kernel void @s8_of_i32_to_f32(ptr addrspace(1) %p, i32 %x) {
  %b = trunc i32 %x to i8
  %f = sitofp i8 %b to float
  store float %f, ptr addrspace(1) %p
  ret void
}

; CHECK-LABEL: s8_to_f16:
; CHECK-NOT:   asr
; CHECK:       convert s8_to_f, {{r[0-9]+l}}, {{r[0-9]+l}}, rte
define agx_kernel void @s8_to_f16(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %b = load i8, ptr addrspace(1) %q
  %f = sitofp i8 %b to half
  store half %f, ptr addrspace(1) %p
  ret void
}
