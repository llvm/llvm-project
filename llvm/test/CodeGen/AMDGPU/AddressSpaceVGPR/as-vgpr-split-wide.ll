; RUN: llc -global-isel=0 -mtriple=amdgpu12.00-- < %s | FileCheck %s
; RUN: llc -global-isel=1 -mtriple=amdgpu12.00-- < %s | FileCheck %s

; Accesses far wider than any VGPR tuple, checked by counting moves: every
; dword is moved once, so no part may be dropped, repeated or wrap onto a
; supported width.

; 2050 dwords is 65600 bits, a width the opcode table cannot key.

; CHECK-LABEL: load_v2050i32:
; CHECK-COUNT-2050: v_movrels_b32_e32
; CHECK-NOT: v_movrels_b32_e32
; CHECK: s_setpc_b64
define <2050 x i32> @load_v2050i32(ptr addrspace(13) inreg %p) {
  %x = load <2050 x i32>, ptr addrspace(13) %p, align 4
  ret <2050 x i32> %x
}

; CHECK-LABEL: store_v2050i32:
; CHECK-COUNT-2050: v_movreld_b32_e32
; CHECK-NOT: v_movreld_b32_e32
; CHECK: s_setpc_b64
define void @store_v2050i32(ptr addrspace(13) inreg %p, ptr addrspace(1) %in) {
  %v = load <2050 x i32>, ptr addrspace(1) %in
  store <2050 x i32> %v, ptr addrspace(13) %p, align 4
  ret void
}

; Each element is wider than any VGPR tuple.
; CHECK-LABEL: load_v2i1056:
; CHECK-COUNT-66: v_movrels_b32_e32
; CHECK-NOT: v_movrels_b32_e32
; CHECK: s_setpc_b64
define <2 x i1056> @load_v2i1056(ptr addrspace(13) inreg %p) {
  %x = load <2 x i1056>, ptr addrspace(13) %p, align 4
  ret <2 x i1056> %x
}

; CHECK-LABEL: store_v2i1056:
; CHECK-COUNT-66: v_movreld_b32_e32
; CHECK-NOT: v_movreld_b32_e32
; CHECK: s_setpc_b64
define void @store_v2i1056(ptr addrspace(13) inreg %p, <2 x i1056> %v) {
  store <2 x i1056> %v, ptr addrspace(13) %p, align 4
  ret void
}
