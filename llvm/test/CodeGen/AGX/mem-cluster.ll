; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs -agx-mem-cluster=0 < %s | FileCheck %s --check-prefix=OFF

; Device stores are kept together: a store issued just after an ALU
; instruction waits for the ALU pipeline to drain, a run of stores once.
define agx_kernel void @stores(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: stores:
; CHECK:         device_load 0, i16, x, [[V:r[0-9]+l]], u0_u1, 0, signed
; CHECK-NEXT:    mov_imm [[C:r[0-9]+[lh]]], 22136
; CHECK-NEXT:    wait 0
; CHECK-NEXT:    device_store 0, i16, x, [[V]], u2_u3, 1, signed
; CHECK-NEXT:    device_store 0, i16, x, [[C]], u2_u3, 0, signed
; CHECK-NEXT:    stop
; OFF-LABEL: stores:
; OFF:         device_store 0, i16, x, {{r[0-9]+l}}, u2_u3, 1, signed
; OFF-NEXT:    mov_imm {{r[0-9]+[lh]}}, 22136
; OFF-NEXT:    device_store 0, i16, x, {{r[0-9]+[lh]}}, u2_u3, 0, signed
  %v = load i16, ptr addrspace(1) %in, align 2
  %w = insertelement <2 x i16> <i16 22136, i16 poison>, i16 %v, i64 1
  store <2 x i16> %w, ptr addrspace(1) %out, align 4
  ret void
}
