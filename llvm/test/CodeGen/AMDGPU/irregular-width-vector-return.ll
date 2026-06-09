; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -o - < %s | FileCheck %s

define <4 x i63> @v4i63_zero() {
; CHECK-LABEL: v4i63_zero:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_mov_b32_e32 v0, 0
; CHECK-NEXT:    v_mov_b32_e32 v1, 0
; CHECK-NEXT:    v_mov_b32_e32 v2, 0
; CHECK-NEXT:    v_mov_b32_e32 v3, 0
; CHECK-NEXT:    v_mov_b32_e32 v4, 0
; CHECK-NEXT:    v_mov_b32_e32 v5, 0
; CHECK-NEXT:    v_mov_b32_e32 v6, 0
; CHECK-NEXT:    v_mov_b32_e32 v7, 0
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  ret <4 x i63> zeroinitializer
}

define <4 x i63> @v4i63_const() {
; CHECK-LABEL: v4i63_const:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CHECK-NEXT:    v_mov_b32_e32 v0, -1
; CHECK-NEXT:    v_not_b32_e32 v1, -2.0
; CHECK-NEXT:    v_mov_b32_e32 v2, 1
; CHECK-NEXT:    v_mov_b32_e32 v3, 0
; CHECK-NEXT:    v_mov_b32_e32 v4, 0
; CHECK-NEXT:    v_bfrev_b32_e32 v5, 4
; CHECK-NEXT:    v_mov_b32_e32 v6, -1
; CHECK-NEXT:    v_bfrev_b32_e32 v7, -2
; CHECK-NEXT:    s_setpc_b64 s[30:31]
  ret <4 x i63> <i63 4611686018427387903, i63 1, i63 2305843009213693952, i63 9223372036854775807>
}

declare <4 x i63> @callee_v4i63()

define <4 x i63> @call_v4i63() {
; CHECK-LABEL: call_v4i63:
; CHECK:       ; %bb.0:
; CHECK:       s_swappc_b64
; CHECK:       s_setpc_b64 s[30:31]
  %v = call <4 x i63> @callee_v4i63()
  ret <4 x i63> %v
}
