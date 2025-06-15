; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -o - < %s | FileCheck --check-prefix=SELDAG %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -o - -global-isel < %s | FileCheck --check-prefix=GISEL %s

; SELDAG-LABEL: test_kill:
; SELDAG-NEXT:  ; %bb.0:
; SELDAG-NEXT:      s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; SELDAG-NEXT:      flat_load_dword v0, v[0:1]
; SELDAG-NEXT:      v_and_b32_e32 v1, 1, v4
; SELDAG-NEXT:      v_cmp_eq_u32_e32 vcc, 1, v1
; SELDAG-NEXT:      s_mov_b64 s[4:5], exec
; SELDAG-NEXT:      s_andn2_b64 s[6:7], exec, vcc
; SELDAG-NEXT:      s_andn2_b64 s[4:5], s[4:5], s[6:7]
; SELDAG-NEXT:      s_cbranch_scc0 .LBB0_2
; SELDAG-NEXT:  ; %bb.1:
; SELDAG-NEXT:      s_and_b64 exec, exec, s[4:5]
; SELDAG-NEXT:      s_waitcnt vmcnt(0) lgkmcnt(0)
; SELDAG-NEXT:      flat_store_dword v[2:3], v0
; SELDAG-NEXT:      s_waitcnt vmcnt(0) lgkmcnt(0)
; SELDAG-NEXT:      s_setpc_b64 s[30:31]
; SELDAG-NEXT:  .LBB0_2:
; SELDAG-NEXT:      s_mov_b64 exec, 0
; SELDAG-NEXT:      s_endpgm

; GISEL-LABEL: test_kill:
; GISEL-NEXT:  ; %bb.0:
; GISEL-NEXT:      s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GISEL-NEXT:      flat_load_dword v0, v[0:1]
; GISEL-NEXT:      v_and_b32_e32 v1, 1, v4
; GISEL-NEXT:      v_cmp_ne_u32_e32 vcc, 0, v1
; GISEL-NEXT:      s_mov_b64 s[4:5], exec
; GISEL-NEXT:      s_andn2_b64 s[6:7], exec, vcc
; GISEL-NEXT:      s_andn2_b64 s[4:5], s[4:5], s[6:7]
; GISEL-NEXT:      s_cbranch_scc0 .LBB0_2
; GISEL-NEXT:  ; %bb.1:
; GISEL-NEXT:      s_and_b64 exec, exec, s[4:5]
; GISEL-NEXT:      s_waitcnt vmcnt(0) lgkmcnt(0)
; GISEL-NEXT:      flat_store_dword v[2:3], v0
; GISEL-NEXT:      s_waitcnt vmcnt(0) lgkmcnt(0)
; GISEL-NEXT:      s_setpc_b64 s[30:31]
; GISEL-NEXT:  .LBB0_2:
; GISEL-NEXT:      s_mov_b64 exec, 0
; GISEL-NEXT:      s_endpgm

define void @test_kill(ptr %src, ptr %dst, i1 %c) {
  %a = load i32, ptr %src, align 4
  callbr void @llvm.amdgcn.kill(i1 %c) to label %cont [label %kill]
kill:
  unreachable
cont:
  store i32 %a, ptr %dst, align 4
  ret void
}

; SELDAG-LABEL: test_kill_block_order:
; SELDAG-NEXT:  ; %bb.0:
; SELDAG-NEXT:      s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; SELDAG-NEXT:      flat_load_dword v0, v[0:1]
; SELDAG-NEXT:      v_and_b32_e32 v1, 1, v4
; SELDAG-NEXT:      v_cmp_eq_u32_e32 vcc, 1, v1
; SELDAG-NEXT:      s_mov_b64 s[4:5], exec
; SELDAG-NEXT:      s_andn2_b64 s[6:7], exec, vcc
; SELDAG-NEXT:      s_andn2_b64 s[4:5], s[4:5], s[6:7]
; SELDAG-NEXT:      s_cbranch_scc0 .LBB1_2
; SELDAG-NEXT:  ; %bb.1:
; SELDAG-NEXT:      s_and_b64 exec, exec, s[4:5]
; SELDAG-NEXT:      s_waitcnt vmcnt(0) lgkmcnt(0)
; SELDAG-NEXT:      flat_store_dword v[2:3], v0
; SELDAG-NEXT:      s_waitcnt vmcnt(0) lgkmcnt(0)
; SELDAG-NEXT:      s_setpc_b64 s[30:31]
; SELDAG-NEXT:  .LBB1_2:
; SELDAG-NEXT:      s_mov_b64 exec, 0
; SELDAG-NEXT:      s_endpgm

; GISEL-LABEL: test_kill_block_order:
; GISEL-NEXT:  ; %bb.0:
; GISEL-NEXT:      s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GISEL-NEXT:      flat_load_dword v0, v[0:1]
; GISEL-NEXT:      v_and_b32_e32 v1, 1, v4
; GISEL-NEXT:      v_cmp_ne_u32_e32 vcc, 0, v1
; GISEL-NEXT:      s_mov_b64 s[4:5], exec
; GISEL-NEXT:      s_andn2_b64 s[6:7], exec, vcc
; GISEL-NEXT:      s_andn2_b64 s[4:5], s[4:5], s[6:7]
; GISEL-NEXT:      s_cbranch_scc0 .LBB1_2
; GISEL-NEXT:  ; %bb.1:
; GISEL-NEXT:      s_and_b64 exec, exec, s[4:5]
; GISEL-NEXT:      s_waitcnt vmcnt(0) lgkmcnt(0)
; GISEL-NEXT:      flat_store_dword v[2:3], v0
; GISEL-NEXT:      s_waitcnt vmcnt(0) lgkmcnt(0)
; GISEL-NEXT:      s_setpc_b64 s[30:31]
; GISEL-NEXT:  .LBB1_2:
; GISEL-NEXT:      s_mov_b64 exec, 0
; GISEL-NEXT:      s_endpgm

define void @test_kill_block_order(ptr %src, ptr %dst, i1 %c) {
  %a = load i32, ptr %src, align 4
  callbr void @llvm.amdgcn.kill(i1 %c) to label %cont [label %kill]
cont:
  store i32 %a, ptr %dst, align 4
  ret void
kill:
  unreachable
}
