; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefix=GFX9 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 < %s | FileCheck -check-prefix=GFX10 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 < %s | FileCheck -check-prefix=GFX11 %s

define <2 x i8> @shuffle_v2i8_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v2i8_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v2
; GFX9-NEXT:                                          ; implicit-def: $vgpr2
; GFX9-NEXT:    s_and_saveexec_b64 s[4:5], vcc
; GFX9-NEXT:    s_cbranch_execz .LBB0_2
; GFX9-NEXT:  ; %bb.1:                                ; %then
; GFX9-NEXT:    global_load_ushort v0, v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_lshrrev_b16_e32 v2, 8, v0
; GFX9-NEXT:  .LBB0_2:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v2
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v2i8_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX10-NEXT:                                          ; implicit-def: $vgpr2
; GFX10-NEXT:    s_and_saveexec_b32 s4, vcc_lo
; GFX10-NEXT:    s_cbranch_execz .LBB0_2
; GFX10-NEXT:  ; %bb.1:                                ; %then
; GFX10-NEXT:    global_load_ushort v0, v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_lshrrev_b16 v2, 8, v0
; GFX10-NEXT:  .LBB0_2:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v2
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v2i8_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX11-NEXT:                                          ; implicit-def: $vgpr2
; GFX11-NEXT:    s_and_saveexec_b32 s0, vcc_lo
; GFX11-NEXT:    s_cbranch_execz .LBB0_2
; GFX11-NEXT:  ; %bb.1:                                ; %then
; GFX11-NEXT:    global_load_u16 v0, v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_lshrrev_b16 v2, 8, v0
; GFX11-NEXT:  .LBB0_2:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v0, v2
; GFX11-NEXT:    v_mov_b32_e32 v1, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <2 x i8>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <2 x i8> %val0, <2 x i8> poison, <2 x i32> <i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <2 x i8> %val0, <2 x i8> poison, <2 x i32> <i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <2 x i8> [ %val1, %then ], [ %val2, %else ]
  ret <2 x i8> %val3
}

define <4 x i8> @shuffle_v4i8_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v4i8_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dword v1, v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB1_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_lshrrev_b32_e32 v0, 16, v1
; GFX9-NEXT:                                          ; implicit-def: $vgpr1
; GFX9-NEXT:  .LBB1_2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB1_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_lshrrev_b32_e32 v0, 8, v1
; GFX9-NEXT:  .LBB1_4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    v_mov_b32_e32 v2, v0
; GFX9-NEXT:    v_mov_b32_e32 v3, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v4i8_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dword v1, v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB1_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_lshrrev_b32_e32 v0, 16, v1
; GFX10-NEXT:                                          ; implicit-def: $vgpr1
; GFX10-NEXT:  .LBB1_2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB1_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_lshrrev_b32_e32 v0, 8, v1
; GFX10-NEXT:  .LBB1_4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    v_mov_b32_e32 v2, v0
; GFX10-NEXT:    v_mov_b32_e32 v3, v0
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v4i8_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b32 v1, v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB1_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_lshrrev_b32_e32 v0, 16, v1
; GFX11-NEXT:                                          ; implicit-def: $vgpr1
; GFX11-NEXT:  .LBB1_2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB1_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_lshrrev_b32_e32 v0, 8, v1
; GFX11-NEXT:  .LBB1_4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    v_mov_b32_e32 v2, v0
; GFX11-NEXT:    v_mov_b32_e32 v3, v0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <4 x i8>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <4 x i8> %val0, <4 x i8> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <4 x i8> %val0, <4 x i8> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <4 x i8> [ %val1, %then ], [ %val2, %else ]
  ret <4 x i8> %val3
}

define <8 x i8> @shuffle_v8i8_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v8i8_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx2 v[0:1], v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v1, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v1
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr3
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    v_lshrrev_b32_e32 v3, 16, v0
; GFX9-NEXT:                                          ; implicit-def: $vgpr0_vgpr1
; GFX9-NEXT:  ; %bb.2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    v_lshrrev_b32_e32 v3, 8, v0
; GFX9-NEXT:  ; %bb.4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_lshlrev_b16_e32 v4, 8, v3
; GFX9-NEXT:    v_or_b32_sdwa v8, v3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b16_e32 v4, 8, v3
; GFX9-NEXT:    v_lshlrev_b16_e32 v2, 8, v3
; GFX9-NEXT:    v_or_b32_sdwa v6, v3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_sdwa v2, v3, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b32_e32 v7, 16, v6
; GFX9-NEXT:    v_lshlrev_b16_e32 v1, 8, v3
; GFX9-NEXT:    v_lshlrev_b32_e32 v9, 16, v2
; GFX9-NEXT:    v_or_b32_sdwa v10, v8, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_sdwa v0, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_e32 v1, v1, v9
; GFX9-NEXT:    v_lshrrev_b64 v[3:4], 24, v[9:10]
; GFX9-NEXT:    v_lshrrev_b32_e32 v5, 8, v10
; GFX9-NEXT:    v_lshrrev_b32_e32 v7, 24, v7
; GFX9-NEXT:    v_lshrrev_b32_e32 v1, 8, v1
; GFX9-NEXT:    v_mov_b32_e32 v4, v8
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v8i8_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx2 v[0:1], v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v1, 1, v2
; GFX10-NEXT:                                          ; implicit-def: $vgpr3
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v1
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    v_lshrrev_b32_e32 v3, 16, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0_vgpr1
; GFX10-NEXT:  ; %bb.2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    v_lshrrev_b32_e32 v3, 8, v0
; GFX10-NEXT:  ; %bb.4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_lshlrev_b16 v0, 8, v3
; GFX10-NEXT:    v_lshlrev_b16 v1, 8, v3
; GFX10-NEXT:    v_lshlrev_b16 v4, 8, v3
; GFX10-NEXT:    v_or_b32_sdwa v6, v3, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v2, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v8, v3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshlrev_b16 v0, 8, v3
; GFX10-NEXT:    v_lshlrev_b32_e32 v1, 16, v6
; GFX10-NEXT:    v_lshlrev_b32_e32 v9, 16, v2
; GFX10-NEXT:    v_or_b32_sdwa v10, v8, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v11, v0, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v0, v3, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshrrev_b32_e32 v7, 24, v1
; GFX10-NEXT:    v_lshrrev_b64 v[3:4], 24, v[9:10]
; GFX10-NEXT:    v_lshrrev_b32_e32 v5, 8, v10
; GFX10-NEXT:    v_lshrrev_b32_e32 v1, 8, v11
; GFX10-NEXT:    v_mov_b32_e32 v4, v8
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v8i8_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b64 v[0:1], v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v1, 1, v2
; GFX11-NEXT:                                          ; implicit-def: $vgpr3
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v1
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    v_lshrrev_b32_e32 v3, 16, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0_vgpr1
; GFX11-NEXT:  ; %bb.2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    v_lshrrev_b32_e32 v3, 8, v0
; GFX11-NEXT:  ; %bb.4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_and_b32_e32 v4, 0xff, v3
; GFX11-NEXT:    v_lshlrev_b16 v5, 8, v3
; GFX11-NEXT:    v_and_b32_e32 v6, 0xff, v3
; GFX11-NEXT:    v_lshlrev_b16 v7, 8, v3
; GFX11-NEXT:    v_and_b32_e32 v0, 0xff, v3
; GFX11-NEXT:    v_lshlrev_b16 v1, 8, v3
; GFX11-NEXT:    v_or_b32_e32 v8, v4, v5
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
; GFX11-NEXT:    v_or_b32_e32 v6, v6, v7
; GFX11-NEXT:    v_or_b32_e32 v2, v0, v1
; GFX11-NEXT:    v_lshlrev_b16 v0, 8, v3
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
; GFX11-NEXT:    v_and_b32_e32 v1, 0xffff, v8
; GFX11-NEXT:    v_lshlrev_b32_e32 v7, 16, v6
; GFX11-NEXT:    v_and_b32_e32 v3, 0xff, v3
; GFX11-NEXT:    v_lshlrev_b32_e32 v4, 16, v2
; GFX11-NEXT:    v_and_b32_e32 v9, 0xffff, v0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
; GFX11-NEXT:    v_or_b32_e32 v5, v1, v7
; GFX11-NEXT:    v_or_b32_e32 v0, v3, v0
; GFX11-NEXT:    v_lshrrev_b32_e32 v7, 24, v7
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
; GFX11-NEXT:    v_or_b32_e32 v1, v9, v4
; GFX11-NEXT:    v_lshrrev_b64 v[3:4], 24, v[4:5]
; GFX11-NEXT:    v_mov_b32_e32 v4, v8
; GFX11-NEXT:    v_lshrrev_b32_e32 v5, 8, v5
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_4)
; GFX11-NEXT:    v_lshrrev_b32_e32 v1, 8, v1
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <8 x i8>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <8 x i8> %val0, <8 x i8> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <8 x i8> %val0, <8 x i8> poison, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <8 x i8> [ %val1, %then ], [ %val2, %else ]
  ret <8 x i8> %val3
}

define <16 x i8> @shuffle_v16i8_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v16i8_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr1
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB3_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_lshrrev_b32_e32 v1, 16, v3
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX9-NEXT:  .LBB3_2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB3_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_lshrrev_b32_e32 v1, 8, v3
; GFX9-NEXT:  .LBB3_4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_lshlrev_b16_e32 v2, 8, v1
; GFX9-NEXT:    v_lshlrev_b16_e32 v10, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v2, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_sdwa v10, v1, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_lshlrev_b16_e32 v4, 8, v1
; GFX9-NEXT:    v_lshlrev_b32_e32 v3, 16, v2
; GFX9-NEXT:    v_lshlrev_b16_e32 v12, 8, v1
; GFX9-NEXT:    v_lshlrev_b32_e32 v11, 16, v10
; GFX9-NEXT:    v_or_b32_sdwa v0, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_e32 v9, v4, v3
; GFX9-NEXT:    v_lshlrev_b16_e32 v4, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v8, v1, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_e32 v18, v12, v11
; GFX9-NEXT:    v_lshlrev_b16_e32 v12, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v17, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b16_e32 v4, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v16, v1, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b16_e32 v12, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v6, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_sdwa v14, v1, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b32_e32 v7, 16, v6
; GFX9-NEXT:    v_lshlrev_b32_e32 v1, 16, v14
; GFX9-NEXT:    v_or_b32_sdwa v4, v17, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_sdwa v12, v16, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshrrev_b32_e32 v5, 8, v4
; GFX9-NEXT:    v_lshrrev_b32_e32 v13, 8, v12
; GFX9-NEXT:    v_lshrrev_b64 v[3:4], 24, v[3:4]
; GFX9-NEXT:    v_lshrrev_b64 v[11:12], 24, v[11:12]
; GFX9-NEXT:    v_lshrrev_b32_e32 v7, 24, v7
; GFX9-NEXT:    v_lshrrev_b32_e32 v15, 24, v1
; GFX9-NEXT:    v_lshrrev_b32_e32 v1, 8, v9
; GFX9-NEXT:    v_lshrrev_b32_e32 v9, 8, v18
; GFX9-NEXT:    v_mov_b32_e32 v4, v17
; GFX9-NEXT:    v_mov_b32_e32 v12, v16
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v16i8_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:                                          ; implicit-def: $vgpr1
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB3_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_lshrrev_b32_e32 v1, 16, v3
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX10-NEXT:  .LBB3_2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB3_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_lshrrev_b32_e32 v1, 8, v3
; GFX10-NEXT:  .LBB3_4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_lshlrev_b16 v3, 8, v1
; GFX10-NEXT:    v_lshlrev_b16 v4, 8, v1
; GFX10-NEXT:    v_lshlrev_b16 v2, 8, v1
; GFX10-NEXT:    v_lshlrev_b16 v7, 8, v1
; GFX10-NEXT:    v_lshlrev_b16 v8, 8, v1
; GFX10-NEXT:    v_or_b32_sdwa v6, v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshlrev_b16 v3, 8, v1
; GFX10-NEXT:    v_or_b32_sdwa v14, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v2, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v17, v1, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshlrev_b32_e32 v7, 16, v6
; GFX10-NEXT:    v_or_b32_sdwa v10, v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v16, v1, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshlrev_b32_e32 v9, 16, v14
; GFX10-NEXT:    v_lshlrev_b16 v5, 8, v1
; GFX10-NEXT:    v_lshlrev_b32_e32 v3, 16, v2
; GFX10-NEXT:    v_lshlrev_b16 v13, 8, v1
; GFX10-NEXT:    v_lshlrev_b32_e32 v11, 16, v10
; GFX10-NEXT:    v_or_b32_sdwa v4, v17, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v12, v16, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v15, v5, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v0, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v18, v13, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshrrev_b32_e32 v5, 8, v4
; GFX10-NEXT:    v_or_b32_sdwa v8, v1, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshrrev_b32_e32 v13, 8, v12
; GFX10-NEXT:    v_lshrrev_b64 v[3:4], 24, v[3:4]
; GFX10-NEXT:    v_lshrrev_b64 v[11:12], 24, v[11:12]
; GFX10-NEXT:    v_lshrrev_b32_e32 v1, 8, v15
; GFX10-NEXT:    v_lshrrev_b32_e32 v7, 24, v7
; GFX10-NEXT:    v_lshrrev_b32_e32 v15, 24, v9
; GFX10-NEXT:    v_lshrrev_b32_e32 v9, 8, v18
; GFX10-NEXT:    v_mov_b32_e32 v4, v17
; GFX10-NEXT:    v_mov_b32_e32 v12, v16
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v16i8_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b128 v[3:6], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:                                          ; implicit-def: $vgpr1
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB3_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_lshrrev_b32_e32 v1, 16, v3
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX11-NEXT:  .LBB3_2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB3_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_lshrrev_b32_e32 v1, 8, v3
; GFX11-NEXT:  .LBB3_4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_and_b32_e32 v8, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b16 v9, 8, v1
; GFX11-NEXT:    v_and_b32_e32 v0, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b16 v2, 8, v1
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_lshlrev_b16 v6, 8, v1
; GFX11-NEXT:    v_lshlrev_b16 v7, 8, v1
; GFX11-NEXT:    v_or_b32_e32 v16, v8, v9
; GFX11-NEXT:    v_and_b32_e32 v5, 0xff, v1
; GFX11-NEXT:    v_or_b32_e32 v2, v0, v2
; GFX11-NEXT:    v_and_b32_e32 v3, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b16 v4, 8, v1
; GFX11-NEXT:    v_and_b32_e32 v10, 0xff, v1
; GFX11-NEXT:    v_or_b32_e32 v0, v5, v6
; GFX11-NEXT:    v_and_b32_e32 v5, 0xffff, v6
; GFX11-NEXT:    v_and_b32_e32 v6, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b16 v11, 8, v1
; GFX11-NEXT:    v_lshlrev_b16 v12, 8, v1
; GFX11-NEXT:    v_or_b32_e32 v17, v3, v4
; GFX11-NEXT:    v_lshlrev_b16 v8, 8, v1
; GFX11-NEXT:    v_or_b32_e32 v6, v6, v7
; GFX11-NEXT:    v_and_b32_e32 v7, 0xff, v1
; GFX11-NEXT:    v_or_b32_e32 v14, v10, v11
; GFX11-NEXT:    v_and_b32_e32 v4, 0xffff, v17
; GFX11-NEXT:    v_lshlrev_b32_e32 v3, 16, v2
; GFX11-NEXT:    v_lshlrev_b32_e32 v9, 16, v6
; GFX11-NEXT:    v_or_b32_e32 v10, v7, v12
; GFX11-NEXT:    v_and_b32_e32 v7, 0xffff, v16
; GFX11-NEXT:    v_lshlrev_b32_e32 v15, 16, v14
; GFX11-NEXT:    v_and_b32_e32 v13, 0xffff, v8
; GFX11-NEXT:    v_or_b32_e32 v4, v4, v9
; GFX11-NEXT:    v_lshlrev_b32_e32 v11, 16, v10
; GFX11-NEXT:    v_and_b32_e32 v1, 0xff, v1
; GFX11-NEXT:    v_or_b32_e32 v12, v7, v15
; GFX11-NEXT:    v_or_b32_e32 v7, v5, v3
; GFX11-NEXT:    v_lshrrev_b32_e32 v5, 8, v4
; GFX11-NEXT:    v_or_b32_e32 v18, v13, v11
; GFX11-NEXT:    v_lshrrev_b64 v[3:4], 24, v[3:4]
; GFX11-NEXT:    v_mov_b32_e32 v4, v17
; GFX11-NEXT:    v_lshrrev_b32_e32 v13, 8, v12
; GFX11-NEXT:    v_lshrrev_b64 v[11:12], 24, v[11:12]
; GFX11-NEXT:    v_or_b32_e32 v8, v1, v8
; GFX11-NEXT:    v_lshrrev_b32_e32 v1, 8, v7
; GFX11-NEXT:    v_lshrrev_b32_e32 v7, 24, v9
; GFX11-NEXT:    v_lshrrev_b32_e32 v15, 24, v15
; GFX11-NEXT:    v_lshrrev_b32_e32 v9, 8, v18
; GFX11-NEXT:    v_mov_b32_e32 v12, v16
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <16 x i8>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <16 x i8> %val0, <16 x i8> poison, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <16 x i8> %val0, <16 x i8> poison, <16 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <16 x i8> [ %val1, %then ], [ %val2, %else ]
  ret <16 x i8> %val3
}

define <32 x i8> @shuffle_v32i8_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v32i8_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr1
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB4_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_lshrrev_b32_e32 v1, 16, v3
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX9-NEXT:  .LBB4_2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB4_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_lshrrev_b32_e32 v1, 8, v3
; GFX9-NEXT:  .LBB4_4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_lshlrev_b16_e32 v10, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v10, v1, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_lshlrev_b16_e32 v5, 8, v1
; GFX9-NEXT:    v_lshlrev_b32_e32 v11, 16, v10
; GFX9-NEXT:    v_or_b32_sdwa v8, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_e32 v17, v5, v11
; GFX9-NEXT:    v_lshlrev_b16_e32 v5, 8, v1
; GFX9-NEXT:    v_lshlrev_b16_e32 v13, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v34, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b16_e32 v5, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v18, v1, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_sdwa v14, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b16_e32 v5, 8, v1
; GFX9-NEXT:    v_lshlrev_b32_e32 v19, 16, v18
; GFX9-NEXT:    v_lshlrev_b16_e32 v2, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v16, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_e32 v25, v5, v19
; GFX9-NEXT:    v_lshlrev_b16_e32 v5, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v2, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_sdwa v35, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b16_e32 v5, 8, v1
; GFX9-NEXT:    v_lshlrev_b16_e32 v4, 8, v1
; GFX9-NEXT:    v_lshlrev_b32_e32 v3, 16, v2
; GFX9-NEXT:    v_or_b32_sdwa v22, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b16_e32 v5, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v0, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_e32 v9, v4, v3
; GFX9-NEXT:    v_lshlrev_b16_e32 v4, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v26, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b16_e32 v5, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v33, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b16_e32 v4, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v32, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b16_e32 v5, 8, v1
; GFX9-NEXT:    v_or_b32_sdwa v6, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_sdwa v30, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b16_e32 v5, 8, v1
; GFX9-NEXT:    v_lshlrev_b32_e32 v7, 16, v6
; GFX9-NEXT:    v_lshlrev_b32_e32 v15, 16, v14
; GFX9-NEXT:    v_lshlrev_b32_e32 v23, 16, v22
; GFX9-NEXT:    v_or_b32_sdwa v24, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b32_e32 v1, 16, v30
; GFX9-NEXT:    v_or_b32_sdwa v4, v33, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_sdwa v12, v34, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_sdwa v20, v35, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX9-NEXT:    v_lshlrev_b32_e32 v27, 16, v26
; GFX9-NEXT:    v_or_b32_sdwa v28, v32, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX9-NEXT:    v_or_b32_e32 v36, v5, v27
; GFX9-NEXT:    v_lshrrev_b32_e32 v5, 8, v4
; GFX9-NEXT:    v_lshrrev_b64 v[3:4], 24, v[3:4]
; GFX9-NEXT:    v_lshrrev_b32_e32 v13, 8, v12
; GFX9-NEXT:    v_lshrrev_b64 v[11:12], 24, v[11:12]
; GFX9-NEXT:    v_lshrrev_b32_e32 v21, 8, v20
; GFX9-NEXT:    v_lshrrev_b64 v[19:20], 24, v[19:20]
; GFX9-NEXT:    v_lshrrev_b32_e32 v29, 8, v28
; GFX9-NEXT:    v_lshrrev_b64 v[27:28], 24, v[27:28]
; GFX9-NEXT:    v_lshrrev_b32_e32 v7, 24, v7
; GFX9-NEXT:    v_lshrrev_b32_e32 v15, 24, v15
; GFX9-NEXT:    v_lshrrev_b32_e32 v23, 24, v23
; GFX9-NEXT:    v_lshrrev_b32_e32 v31, 24, v1
; GFX9-NEXT:    v_lshrrev_b32_e32 v1, 8, v9
; GFX9-NEXT:    v_lshrrev_b32_e32 v9, 8, v17
; GFX9-NEXT:    v_lshrrev_b32_e32 v17, 8, v25
; GFX9-NEXT:    v_lshrrev_b32_e32 v25, 8, v36
; GFX9-NEXT:    v_mov_b32_e32 v4, v33
; GFX9-NEXT:    v_mov_b32_e32 v12, v34
; GFX9-NEXT:    v_mov_b32_e32 v20, v35
; GFX9-NEXT:    v_mov_b32_e32 v28, v32
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v32i8_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:                                          ; implicit-def: $vgpr1
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB4_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_lshrrev_b32_e32 v1, 16, v3
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX10-NEXT:  .LBB4_2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB4_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_lshrrev_b32_e32 v1, 8, v3
; GFX10-NEXT:  .LBB4_4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_lshlrev_b16 v0, 8, v1
; GFX10-NEXT:    v_lshlrev_b16 v7, 8, v1
; GFX10-NEXT:    v_lshlrev_b16 v8, 8, v1
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_lshlrev_b16 v4, 8, v1
; GFX10-NEXT:    v_lshlrev_b16 v19, 8, v1
; GFX10-NEXT:    v_or_b32_sdwa v6, v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshlrev_b16 v0, 8, v1
; GFX10-NEXT:    v_or_b32_sdwa v34, v1, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v8, v1, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshlrev_b16 v2, 8, v1
; GFX10-NEXT:    v_lshlrev_b16 v18, 8, v1
; GFX10-NEXT:    v_or_b32_sdwa v10, v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshlrev_b16 v3, 8, v1
; GFX10-NEXT:    v_or_b32_sdwa v14, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshlrev_b16 v20, 8, v1
; GFX10-NEXT:    v_lshlrev_b16 v23, 8, v1
; GFX10-NEXT:    v_lshlrev_b32_e32 v11, 16, v10
; GFX10-NEXT:    v_or_b32_sdwa v30, v1, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v2, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v18, v1, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v33, v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v25, v7, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshlrev_b16 v7, 8, v1
; GFX10-NEXT:    v_lshlrev_b32_e32 v9, 16, v6
; GFX10-NEXT:    v_lshlrev_b32_e32 v15, 16, v14
; GFX10-NEXT:    v_or_b32_sdwa v35, v1, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v32, v1, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v22, v1, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshlrev_b16 v7, 8, v1
; GFX10-NEXT:    v_lshlrev_b32_e32 v36, 16, v30
; GFX10-NEXT:    v_lshlrev_b16 v5, 8, v1
; GFX10-NEXT:    v_lshlrev_b32_e32 v3, 16, v2
; GFX10-NEXT:    v_lshlrev_b32_e32 v31, 16, v22
; GFX10-NEXT:    v_or_b32_sdwa v26, v1, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshlrev_b16 v21, 8, v1
; GFX10-NEXT:    v_lshlrev_b32_e32 v19, 16, v18
; GFX10-NEXT:    v_lshlrev_b16 v7, 8, v1
; GFX10-NEXT:    v_or_b32_sdwa v4, v33, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshlrev_b32_e32 v27, 16, v26
; GFX10-NEXT:    v_or_b32_sdwa v12, v34, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v20, v35, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v28, v32, v36 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v17, v5, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v37, v21, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v38, v7, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; GFX10-NEXT:    v_or_b32_sdwa v0, v1, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshrrev_b32_e32 v5, 8, v4
; GFX10-NEXT:    v_lshrrev_b32_e32 v13, 8, v12
; GFX10-NEXT:    v_or_b32_sdwa v16, v1, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshrrev_b32_e32 v21, 8, v20
; GFX10-NEXT:    v_lshrrev_b32_e32 v29, 8, v28
; GFX10-NEXT:    v_lshrrev_b64 v[3:4], 24, v[3:4]
; GFX10-NEXT:    v_lshrrev_b64 v[11:12], 24, v[11:12]
; GFX10-NEXT:    v_lshrrev_b64 v[19:20], 24, v[19:20]
; GFX10-NEXT:    v_lshrrev_b64 v[27:28], 24, v[27:28]
; GFX10-NEXT:    v_or_b32_sdwa v24, v1, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
; GFX10-NEXT:    v_lshrrev_b32_e32 v7, 24, v9
; GFX10-NEXT:    v_lshrrev_b32_e32 v15, 24, v15
; GFX10-NEXT:    v_lshrrev_b32_e32 v23, 24, v31
; GFX10-NEXT:    v_lshrrev_b32_e32 v31, 24, v36
; GFX10-NEXT:    v_lshrrev_b32_e32 v1, 8, v17
; GFX10-NEXT:    v_lshrrev_b32_e32 v9, 8, v25
; GFX10-NEXT:    v_lshrrev_b32_e32 v17, 8, v37
; GFX10-NEXT:    v_lshrrev_b32_e32 v25, 8, v38
; GFX10-NEXT:    v_mov_b32_e32 v4, v33
; GFX10-NEXT:    v_mov_b32_e32 v12, v34
; GFX10-NEXT:    v_mov_b32_e32 v20, v35
; GFX10-NEXT:    v_mov_b32_e32 v28, v32
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v32i8_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b128 v[3:6], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:                                          ; implicit-def: $vgpr1
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB4_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_lshrrev_b32_e32 v1, 16, v3
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX11-NEXT:  .LBB4_2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB4_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_lshrrev_b32_e32 v1, 8, v3
; GFX11-NEXT:  .LBB4_4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_and_b32_e32 v0, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b16 v2, 8, v1
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_lshlrev_b16 v4, 8, v1
; GFX11-NEXT:    v_and_b32_e32 v3, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b16 v5, 8, v1
; GFX11-NEXT:    v_and_b32_e32 v19, 0xff, v1
; GFX11-NEXT:    v_or_b32_e32 v2, v0, v2
; GFX11-NEXT:    v_lshlrev_b16 v20, 8, v1
; GFX11-NEXT:    v_and_b32_e32 v23, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b16 v24, 8, v1
; GFX11-NEXT:    v_or_b32_e32 v33, v3, v5
; GFX11-NEXT:    v_lshlrev_b16 v8, 8, v1
; GFX11-NEXT:    v_or_b32_e32 v35, v19, v20
; GFX11-NEXT:    v_lshlrev_b32_e32 v3, 16, v2
; GFX11-NEXT:    v_or_b32_e32 v32, v23, v24
; GFX11-NEXT:    v_and_b32_e32 v5, 0xffff, v4
; GFX11-NEXT:    v_and_b32_e32 v10, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b16 v11, 8, v1
; GFX11-NEXT:    v_and_b32_e32 v12, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b16 v13, 8, v1
; GFX11-NEXT:    v_or_b32_e32 v9, v5, v3
; GFX11-NEXT:    v_and_b32_e32 v5, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b16 v15, 8, v1
; GFX11-NEXT:    v_or_b32_e32 v14, v10, v11
; GFX11-NEXT:    v_or_b32_e32 v10, v12, v13
; GFX11-NEXT:    v_lshlrev_b16 v16, 8, v1
; GFX11-NEXT:    v_or_b32_e32 v34, v5, v8
; GFX11-NEXT:    v_and_b32_e32 v8, 0xff, v1
; GFX11-NEXT:    v_and_b32_e32 v13, 0xffff, v15
; GFX11-NEXT:    v_and_b32_e32 v21, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b16 v22, 8, v1
; GFX11-NEXT:    v_and_b32_e32 v12, 0xffff, v34
; GFX11-NEXT:    v_or_b32_e32 v8, v8, v15
; GFX11-NEXT:    v_and_b32_e32 v15, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b32_e32 v17, 16, v14
; GFX11-NEXT:    v_and_b32_e32 v0, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b16 v6, 8, v1
; GFX11-NEXT:    v_and_b32_e32 v26, 0xff, v1
; GFX11-NEXT:    v_or_b32_e32 v18, v15, v16
; GFX11-NEXT:    v_or_b32_e32 v16, v21, v22
; GFX11-NEXT:    v_and_b32_e32 v15, 0xffff, v22
; GFX11-NEXT:    v_and_b32_e32 v21, 0xff, v1
; GFX11-NEXT:    v_lshlrev_b16 v22, 8, v1
; GFX11-NEXT:    v_lshlrev_b16 v27, 8, v1
; GFX11-NEXT:    v_lshlrev_b16 v28, 8, v1
; GFX11-NEXT:    v_lshlrev_b32_e32 v11, 16, v10
; GFX11-NEXT:    v_or_b32_e32 v12, v12, v17
; GFX11-NEXT:    v_or_b32_e32 v22, v21, v22
; GFX11-NEXT:    v_and_b32_e32 v21, 0xff, v1
; GFX11-NEXT:    v_and_b32_e32 v7, 0xff, v1
; GFX11-NEXT:    v_or_b32_e32 v6, v0, v6
; GFX11-NEXT:    v_or_b32_e32 v30, v26, v27
; GFX11-NEXT:    v_lshlrev_b16 v24, 8, v1
; GFX11-NEXT:    v_or_b32_e32 v26, v21, v28
; GFX11-NEXT:    v_or_b32_e32 v25, v13, v11
; GFX11-NEXT:    v_lshrrev_b32_e32 v13, 8, v12
; GFX11-NEXT:    v_and_b32_e32 v20, 0xffff, v35
; GFX11-NEXT:    v_lshlrev_b32_e32 v23, 16, v22
; GFX11-NEXT:    v_lshrrev_b64 v[11:12], 24, v[11:12]
; GFX11-NEXT:    v_mov_b32_e32 v12, v34
; GFX11-NEXT:    v_or_b32_e32 v0, v7, v4
; GFX11-NEXT:    v_and_b32_e32 v4, 0xffff, v33
; GFX11-NEXT:    v_lshlrev_b32_e32 v7, 16, v6
; GFX11-NEXT:    v_and_b32_e32 v21, 0xffff, v32
; GFX11-NEXT:    v_lshlrev_b32_e32 v31, 16, v30
; GFX11-NEXT:    v_lshlrev_b32_e32 v19, 16, v18
; GFX11-NEXT:    v_lshlrev_b32_e32 v27, 16, v26
; GFX11-NEXT:    v_and_b32_e32 v29, 0xffff, v24
; GFX11-NEXT:    v_or_b32_e32 v20, v20, v23
; GFX11-NEXT:    v_or_b32_e32 v4, v4, v7
; GFX11-NEXT:    v_or_b32_e32 v28, v21, v31
; GFX11-NEXT:    v_and_b32_e32 v1, 0xff, v1
; GFX11-NEXT:    v_or_b32_e32 v36, v15, v19
; GFX11-NEXT:    v_or_b32_e32 v37, v29, v27
; GFX11-NEXT:    v_lshrrev_b32_e32 v21, 8, v20
; GFX11-NEXT:    v_lshrrev_b64 v[19:20], 24, v[19:20]
; GFX11-NEXT:    v_lshrrev_b32_e32 v5, 8, v4
; GFX11-NEXT:    v_lshrrev_b32_e32 v29, 8, v28
; GFX11-NEXT:    v_lshrrev_b64 v[27:28], 24, v[27:28]
; GFX11-NEXT:    v_mov_b32_e32 v20, v35
; GFX11-NEXT:    v_lshrrev_b64 v[3:4], 24, v[3:4]
; GFX11-NEXT:    v_or_b32_e32 v24, v1, v24
; GFX11-NEXT:    v_lshrrev_b32_e32 v7, 24, v7
; GFX11-NEXT:    v_lshrrev_b32_e32 v15, 24, v17
; GFX11-NEXT:    v_lshrrev_b32_e32 v23, 24, v23
; GFX11-NEXT:    v_lshrrev_b32_e32 v31, 24, v31
; GFX11-NEXT:    v_lshrrev_b32_e32 v1, 8, v9
; GFX11-NEXT:    v_lshrrev_b32_e32 v9, 8, v25
; GFX11-NEXT:    v_lshrrev_b32_e32 v17, 8, v36
; GFX11-NEXT:    v_lshrrev_b32_e32 v25, 8, v37
; GFX11-NEXT:    v_mov_b32_e32 v4, v33
; GFX11-NEXT:    v_mov_b32_e32 v28, v32
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <32 x i8>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <32 x i8> %val0, <32 x i8> poison, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <32 x i8> %val0, <32 x i8> poison, <32 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <32 x i8> [ %val1, %then ], [ %val2, %else ]
  ret <32 x i8> %val3
}

define <2 x i16> @shuffle_v2i16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v2i16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v2
; GFX9-NEXT:                                          ; implicit-def: $vgpr2
; GFX9-NEXT:    s_and_saveexec_b64 s[4:5], vcc
; GFX9-NEXT:    s_cbranch_execz .LBB5_2
; GFX9-NEXT:  ; %bb.1:                                ; %then
; GFX9-NEXT:    global_load_dword v0, v[0:1], off
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v2, v0, v0, s6
; GFX9-NEXT:  .LBB5_2:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v2i16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX10-NEXT:                                          ; implicit-def: $vgpr2
; GFX10-NEXT:    s_and_saveexec_b32 s4, vcc_lo
; GFX10-NEXT:    s_cbranch_execz .LBB5_2
; GFX10-NEXT:  ; %bb.1:                                ; %then
; GFX10-NEXT:    global_load_dword v0, v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v2, v0, v0, 0x7060302
; GFX10-NEXT:  .LBB5_2:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v2i16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX11-NEXT:                                          ; implicit-def: $vgpr2
; GFX11-NEXT:    s_and_saveexec_b32 s0, vcc_lo
; GFX11-NEXT:    s_cbranch_execz .LBB5_2
; GFX11-NEXT:  ; %bb.1:                                ; %then
; GFX11-NEXT:    global_load_b32 v0, v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v2, v0, v0, 0x7060302
; GFX11-NEXT:  .LBB5_2:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v0, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <2 x i16>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <2 x i16> %val0, <2 x i16> poison, <2 x i32> <i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <2 x i16> %val0, <2 x i16> poison, <2 x i32> <i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <2 x i16> [ %val1, %then ], [ %val2, %else ]
  ret <2 x i16> %val3
}

define <4 x i16> @shuffle_v4i16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v4i16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx2 v[3:4], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB6_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4
; GFX9-NEXT:  .LBB6_2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB6_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB6_4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v4i16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx2 v[3:4], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB6_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4
; GFX10-NEXT:  .LBB6_2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB6_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB6_4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v4i16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b64 v[3:4], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB6_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4
; GFX11-NEXT:  .LBB6_2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB6_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB6_4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <4 x i16>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <4 x i16> %val0, <4 x i16> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <4 x i16> %val0, <4 x i16> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <4 x i16> [ %val1, %then ], [ %val2, %else ]
  ret <4 x i16> %val3
}

define <8 x i16> @shuffle_v8i16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v8i16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB7_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX9-NEXT:  .LBB7_2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB7_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB7_4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    v_mov_b32_e32 v2, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v3, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v8i16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB7_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX10-NEXT:  .LBB7_2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB7_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB7_4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    v_mov_b32_e32 v2, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v3, v0
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v8i16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b128 v[3:6], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB7_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX11-NEXT:  .LBB7_2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB7_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB7_4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    v_mov_b32_e32 v2, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v3, v0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <8 x i16>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <8 x i16> %val0, <8 x i16> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <8 x i16> %val0, <8 x i16> poison, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <8 x i16> [ %val1, %then ], [ %val2, %else ]
  ret <8 x i16> %val3
}

define <16 x i16> @shuffle_v16i16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v16i16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB8_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX9-NEXT:  .LBB8_2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB8_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB8_4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    v_mov_b32_e32 v2, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v3, v0
; GFX9-NEXT:    v_mov_b32_e32 v4, v0
; GFX9-NEXT:    v_mov_b32_e32 v5, v0
; GFX9-NEXT:    v_mov_b32_e32 v6, v0
; GFX9-NEXT:    v_mov_b32_e32 v7, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v16i16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB8_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX10-NEXT:  .LBB8_2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB8_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB8_4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    v_mov_b32_e32 v2, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v3, v0
; GFX10-NEXT:    v_mov_b32_e32 v4, v0
; GFX10-NEXT:    v_mov_b32_e32 v5, v0
; GFX10-NEXT:    v_mov_b32_e32 v6, v0
; GFX10-NEXT:    v_mov_b32_e32 v7, v0
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v16i16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b128 v[3:6], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB8_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX11-NEXT:  .LBB8_2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB8_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB8_4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    v_mov_b32_e32 v2, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v3, v0
; GFX11-NEXT:    v_mov_b32_e32 v4, v0
; GFX11-NEXT:    v_mov_b32_e32 v5, v0
; GFX11-NEXT:    v_mov_b32_e32 v6, v0
; GFX11-NEXT:    v_mov_b32_e32 v7, v0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <16 x i16>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <16 x i16> %val0, <16 x i16> poison, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <16 x i16> %val0, <16 x i16> poison, <16 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <16 x i16> [ %val1, %then ], [ %val2, %else ]
  ret <16 x i16> %val3
}

define <32 x i16> @shuffle_v32i16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v32i16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB9_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX9-NEXT:  .LBB9_2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB9_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB9_4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    v_mov_b32_e32 v2, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v3, v0
; GFX9-NEXT:    v_mov_b32_e32 v4, v0
; GFX9-NEXT:    v_mov_b32_e32 v5, v0
; GFX9-NEXT:    v_mov_b32_e32 v6, v0
; GFX9-NEXT:    v_mov_b32_e32 v7, v0
; GFX9-NEXT:    v_mov_b32_e32 v8, v0
; GFX9-NEXT:    v_mov_b32_e32 v9, v0
; GFX9-NEXT:    v_mov_b32_e32 v10, v0
; GFX9-NEXT:    v_mov_b32_e32 v11, v0
; GFX9-NEXT:    v_mov_b32_e32 v12, v0
; GFX9-NEXT:    v_mov_b32_e32 v13, v0
; GFX9-NEXT:    v_mov_b32_e32 v14, v0
; GFX9-NEXT:    v_mov_b32_e32 v15, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v32i16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB9_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX10-NEXT:  .LBB9_2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB9_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB9_4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    v_mov_b32_e32 v2, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v3, v0
; GFX10-NEXT:    v_mov_b32_e32 v4, v0
; GFX10-NEXT:    v_mov_b32_e32 v5, v0
; GFX10-NEXT:    v_mov_b32_e32 v6, v0
; GFX10-NEXT:    v_mov_b32_e32 v7, v0
; GFX10-NEXT:    v_mov_b32_e32 v8, v0
; GFX10-NEXT:    v_mov_b32_e32 v9, v0
; GFX10-NEXT:    v_mov_b32_e32 v10, v0
; GFX10-NEXT:    v_mov_b32_e32 v11, v0
; GFX10-NEXT:    v_mov_b32_e32 v12, v0
; GFX10-NEXT:    v_mov_b32_e32 v13, v0
; GFX10-NEXT:    v_mov_b32_e32 v14, v0
; GFX10-NEXT:    v_mov_b32_e32 v15, v0
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v32i16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b128 v[3:6], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB9_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX11-NEXT:  .LBB9_2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB9_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB9_4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    v_mov_b32_e32 v2, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v3, v0
; GFX11-NEXT:    v_mov_b32_e32 v4, v0
; GFX11-NEXT:    v_mov_b32_e32 v5, v0
; GFX11-NEXT:    v_mov_b32_e32 v6, v0
; GFX11-NEXT:    v_mov_b32_e32 v7, v0
; GFX11-NEXT:    v_mov_b32_e32 v8, v0
; GFX11-NEXT:    v_mov_b32_e32 v9, v0
; GFX11-NEXT:    v_mov_b32_e32 v10, v0
; GFX11-NEXT:    v_mov_b32_e32 v11, v0
; GFX11-NEXT:    v_mov_b32_e32 v12, v0
; GFX11-NEXT:    v_mov_b32_e32 v13, v0
; GFX11-NEXT:    v_mov_b32_e32 v14, v0
; GFX11-NEXT:    v_mov_b32_e32 v15, v0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <32 x i16>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <32 x i16> %val0, <32 x i16> poison, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <32 x i16> %val0, <32 x i16> poison, <32 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <32 x i16> [ %val1, %then ], [ %val2, %else ]
  ret <32 x i16> %val3
}

define <2 x i32> @shuffle_v2i32_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v2i32_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v2
; GFX9-NEXT:                                          ; implicit-def: $vgpr2
; GFX9-NEXT:    s_and_saveexec_b64 s[4:5], vcc
; GFX9-NEXT:    s_cbranch_execz .LBB10_2
; GFX9-NEXT:  ; %bb.1:                                ; %then
; GFX9-NEXT:    global_load_dword v2, v[0:1], off offset:4
; GFX9-NEXT:  .LBB10_2:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v0, v2
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v2i32_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX10-NEXT:                                          ; implicit-def: $vgpr2
; GFX10-NEXT:    s_and_saveexec_b32 s4, vcc_lo
; GFX10-NEXT:    s_cbranch_execz .LBB10_2
; GFX10-NEXT:  ; %bb.1:                                ; %then
; GFX10-NEXT:    global_load_dword v2, v[0:1], off offset:4
; GFX10-NEXT:  .LBB10_2:                               ; %finally
; GFX10-NEXT:    s_waitcnt_depctr 0xffe3
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v0, v2
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v2i32_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX11-NEXT:                                          ; implicit-def: $vgpr2
; GFX11-NEXT:    s_and_saveexec_b32 s0, vcc_lo
; GFX11-NEXT:    s_cbranch_execz .LBB10_2
; GFX11-NEXT:  ; %bb.1:                                ; %then
; GFX11-NEXT:    global_load_b32 v2, v[0:1], off offset:4
; GFX11-NEXT:  .LBB10_2:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v0, v2
; GFX11-NEXT:    v_mov_b32_e32 v1, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <2 x i32>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <2 x i32> %val0, <2 x i32> poison, <2 x i32> <i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <2 x i32> %val0, <2 x i32> poison, <2 x i32> <i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <2 x i32> [ %val1, %then ], [ %val2, %else ]
  ret <2 x i32> %val3
}

define <4 x i32> @shuffle_v4i32_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v4i32_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:  ; %bb.2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    v_mov_b32_e32 v2, v1
; GFX9-NEXT:  ; %bb.4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v1
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:    v_mov_b32_e32 v3, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v4i32_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:  ; %bb.2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    v_mov_b32_e32 v2, v1
; GFX10-NEXT:  ; %bb.4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v1
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:    v_mov_b32_e32 v3, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v4i32_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    v_mov_b32_e32 v1, v2
; GFX11-NEXT:  ; %bb.2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v2, v1
; GFX11-NEXT:  ; %bb.4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_dual_mov_b32 v0, v1 :: v_dual_mov_b32 v1, v2
; GFX11-NEXT:    v_mov_b32_e32 v3, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <4 x i32>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <4 x i32> %val0, <4 x i32> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <4 x i32> %val0, <4 x i32> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <4 x i32> [ %val1, %then ], [ %val2, %else ]
  ret <4 x i32> %val3
}

define <8 x i32> @shuffle_v8i32_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v8i32_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:  ; %bb.2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    v_mov_b32_e32 v2, v1
; GFX9-NEXT:  ; %bb.4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v1
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:    v_mov_b32_e32 v3, v2
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    v_mov_b32_e32 v5, v2
; GFX9-NEXT:    v_mov_b32_e32 v6, v2
; GFX9-NEXT:    v_mov_b32_e32 v7, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v8i32_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:  ; %bb.2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    v_mov_b32_e32 v2, v1
; GFX10-NEXT:  ; %bb.4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v1
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:    v_mov_b32_e32 v3, v2
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    v_mov_b32_e32 v5, v2
; GFX10-NEXT:    v_mov_b32_e32 v6, v2
; GFX10-NEXT:    v_mov_b32_e32 v7, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v8i32_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    v_mov_b32_e32 v1, v2
; GFX11-NEXT:  ; %bb.2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v2, v1
; GFX11-NEXT:  ; %bb.4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_dual_mov_b32 v0, v1 :: v_dual_mov_b32 v1, v2
; GFX11-NEXT:    v_mov_b32_e32 v3, v2
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    v_mov_b32_e32 v5, v2
; GFX11-NEXT:    v_mov_b32_e32 v6, v2
; GFX11-NEXT:    v_mov_b32_e32 v7, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <8 x i32>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <8 x i32> %val0, <8 x i32> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <8 x i32> %val0, <8 x i32> poison, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <8 x i32> [ %val1, %then ], [ %val2, %else ]
  ret <8 x i32> %val3
}

define <16 x i32> @shuffle_v16i32_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v16i32_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:  ; %bb.2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    v_mov_b32_e32 v2, v1
; GFX9-NEXT:  ; %bb.4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v1
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:    v_mov_b32_e32 v3, v2
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    v_mov_b32_e32 v5, v2
; GFX9-NEXT:    v_mov_b32_e32 v6, v2
; GFX9-NEXT:    v_mov_b32_e32 v7, v2
; GFX9-NEXT:    v_mov_b32_e32 v8, v2
; GFX9-NEXT:    v_mov_b32_e32 v9, v2
; GFX9-NEXT:    v_mov_b32_e32 v10, v2
; GFX9-NEXT:    v_mov_b32_e32 v11, v2
; GFX9-NEXT:    v_mov_b32_e32 v12, v2
; GFX9-NEXT:    v_mov_b32_e32 v13, v2
; GFX9-NEXT:    v_mov_b32_e32 v14, v2
; GFX9-NEXT:    v_mov_b32_e32 v15, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v16i32_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:  ; %bb.2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    v_mov_b32_e32 v2, v1
; GFX10-NEXT:  ; %bb.4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v1
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:    v_mov_b32_e32 v3, v2
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    v_mov_b32_e32 v5, v2
; GFX10-NEXT:    v_mov_b32_e32 v6, v2
; GFX10-NEXT:    v_mov_b32_e32 v7, v2
; GFX10-NEXT:    v_mov_b32_e32 v8, v2
; GFX10-NEXT:    v_mov_b32_e32 v9, v2
; GFX10-NEXT:    v_mov_b32_e32 v10, v2
; GFX10-NEXT:    v_mov_b32_e32 v11, v2
; GFX10-NEXT:    v_mov_b32_e32 v12, v2
; GFX10-NEXT:    v_mov_b32_e32 v13, v2
; GFX10-NEXT:    v_mov_b32_e32 v14, v2
; GFX10-NEXT:    v_mov_b32_e32 v15, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v16i32_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    v_mov_b32_e32 v1, v2
; GFX11-NEXT:  ; %bb.2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v2, v1
; GFX11-NEXT:  ; %bb.4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_dual_mov_b32 v0, v1 :: v_dual_mov_b32 v1, v2
; GFX11-NEXT:    v_mov_b32_e32 v3, v2
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    v_mov_b32_e32 v5, v2
; GFX11-NEXT:    v_mov_b32_e32 v6, v2
; GFX11-NEXT:    v_mov_b32_e32 v7, v2
; GFX11-NEXT:    v_mov_b32_e32 v8, v2
; GFX11-NEXT:    v_mov_b32_e32 v9, v2
; GFX11-NEXT:    v_mov_b32_e32 v10, v2
; GFX11-NEXT:    v_mov_b32_e32 v11, v2
; GFX11-NEXT:    v_mov_b32_e32 v12, v2
; GFX11-NEXT:    v_mov_b32_e32 v13, v2
; GFX11-NEXT:    v_mov_b32_e32 v14, v2
; GFX11-NEXT:    v_mov_b32_e32 v15, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <16 x i32>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <16 x i32> %val0, <16 x i32> poison, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <16 x i32> %val0, <16 x i32> poison, <16 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <16 x i32> [ %val1, %then ], [ %val2, %else ]
  ret <16 x i32> %val3
}

define <32 x i32> @shuffle_v32i32_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v32i32_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:  ; %bb.2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    v_mov_b32_e32 v2, v1
; GFX9-NEXT:  ; %bb.4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v1
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:    v_mov_b32_e32 v3, v2
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    v_mov_b32_e32 v5, v2
; GFX9-NEXT:    v_mov_b32_e32 v6, v2
; GFX9-NEXT:    v_mov_b32_e32 v7, v2
; GFX9-NEXT:    v_mov_b32_e32 v8, v2
; GFX9-NEXT:    v_mov_b32_e32 v9, v2
; GFX9-NEXT:    v_mov_b32_e32 v10, v2
; GFX9-NEXT:    v_mov_b32_e32 v11, v2
; GFX9-NEXT:    v_mov_b32_e32 v12, v2
; GFX9-NEXT:    v_mov_b32_e32 v13, v2
; GFX9-NEXT:    v_mov_b32_e32 v14, v2
; GFX9-NEXT:    v_mov_b32_e32 v15, v2
; GFX9-NEXT:    v_mov_b32_e32 v16, v2
; GFX9-NEXT:    v_mov_b32_e32 v17, v2
; GFX9-NEXT:    v_mov_b32_e32 v18, v2
; GFX9-NEXT:    v_mov_b32_e32 v19, v2
; GFX9-NEXT:    v_mov_b32_e32 v20, v2
; GFX9-NEXT:    v_mov_b32_e32 v21, v2
; GFX9-NEXT:    v_mov_b32_e32 v22, v2
; GFX9-NEXT:    v_mov_b32_e32 v23, v2
; GFX9-NEXT:    v_mov_b32_e32 v24, v2
; GFX9-NEXT:    v_mov_b32_e32 v25, v2
; GFX9-NEXT:    v_mov_b32_e32 v26, v2
; GFX9-NEXT:    v_mov_b32_e32 v27, v2
; GFX9-NEXT:    v_mov_b32_e32 v28, v2
; GFX9-NEXT:    v_mov_b32_e32 v29, v2
; GFX9-NEXT:    v_mov_b32_e32 v30, v2
; GFX9-NEXT:    v_mov_b32_e32 v31, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v32i32_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:  ; %bb.2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    v_mov_b32_e32 v2, v1
; GFX10-NEXT:  ; %bb.4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v1
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:    v_mov_b32_e32 v3, v2
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    v_mov_b32_e32 v5, v2
; GFX10-NEXT:    v_mov_b32_e32 v6, v2
; GFX10-NEXT:    v_mov_b32_e32 v7, v2
; GFX10-NEXT:    v_mov_b32_e32 v8, v2
; GFX10-NEXT:    v_mov_b32_e32 v9, v2
; GFX10-NEXT:    v_mov_b32_e32 v10, v2
; GFX10-NEXT:    v_mov_b32_e32 v11, v2
; GFX10-NEXT:    v_mov_b32_e32 v12, v2
; GFX10-NEXT:    v_mov_b32_e32 v13, v2
; GFX10-NEXT:    v_mov_b32_e32 v14, v2
; GFX10-NEXT:    v_mov_b32_e32 v15, v2
; GFX10-NEXT:    v_mov_b32_e32 v16, v2
; GFX10-NEXT:    v_mov_b32_e32 v17, v2
; GFX10-NEXT:    v_mov_b32_e32 v18, v2
; GFX10-NEXT:    v_mov_b32_e32 v19, v2
; GFX10-NEXT:    v_mov_b32_e32 v20, v2
; GFX10-NEXT:    v_mov_b32_e32 v21, v2
; GFX10-NEXT:    v_mov_b32_e32 v22, v2
; GFX10-NEXT:    v_mov_b32_e32 v23, v2
; GFX10-NEXT:    v_mov_b32_e32 v24, v2
; GFX10-NEXT:    v_mov_b32_e32 v25, v2
; GFX10-NEXT:    v_mov_b32_e32 v26, v2
; GFX10-NEXT:    v_mov_b32_e32 v27, v2
; GFX10-NEXT:    v_mov_b32_e32 v28, v2
; GFX10-NEXT:    v_mov_b32_e32 v29, v2
; GFX10-NEXT:    v_mov_b32_e32 v30, v2
; GFX10-NEXT:    v_mov_b32_e32 v31, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v32i32_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    v_mov_b32_e32 v1, v2
; GFX11-NEXT:  ; %bb.2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v2, v1
; GFX11-NEXT:  ; %bb.4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_dual_mov_b32 v0, v1 :: v_dual_mov_b32 v1, v2
; GFX11-NEXT:    v_mov_b32_e32 v3, v2
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    v_mov_b32_e32 v5, v2
; GFX11-NEXT:    v_mov_b32_e32 v6, v2
; GFX11-NEXT:    v_mov_b32_e32 v7, v2
; GFX11-NEXT:    v_mov_b32_e32 v8, v2
; GFX11-NEXT:    v_mov_b32_e32 v9, v2
; GFX11-NEXT:    v_mov_b32_e32 v10, v2
; GFX11-NEXT:    v_mov_b32_e32 v11, v2
; GFX11-NEXT:    v_mov_b32_e32 v12, v2
; GFX11-NEXT:    v_mov_b32_e32 v13, v2
; GFX11-NEXT:    v_mov_b32_e32 v14, v2
; GFX11-NEXT:    v_mov_b32_e32 v15, v2
; GFX11-NEXT:    v_mov_b32_e32 v16, v2
; GFX11-NEXT:    v_mov_b32_e32 v17, v2
; GFX11-NEXT:    v_mov_b32_e32 v18, v2
; GFX11-NEXT:    v_mov_b32_e32 v19, v2
; GFX11-NEXT:    v_mov_b32_e32 v20, v2
; GFX11-NEXT:    v_mov_b32_e32 v21, v2
; GFX11-NEXT:    v_mov_b32_e32 v22, v2
; GFX11-NEXT:    v_mov_b32_e32 v23, v2
; GFX11-NEXT:    v_mov_b32_e32 v24, v2
; GFX11-NEXT:    v_mov_b32_e32 v25, v2
; GFX11-NEXT:    v_mov_b32_e32 v26, v2
; GFX11-NEXT:    v_mov_b32_e32 v27, v2
; GFX11-NEXT:    v_mov_b32_e32 v28, v2
; GFX11-NEXT:    v_mov_b32_e32 v29, v2
; GFX11-NEXT:    v_mov_b32_e32 v30, v2
; GFX11-NEXT:    v_mov_b32_e32 v31, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <32 x i32>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <32 x i32> %val0, <32 x i32> poison, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <32 x i32> %val0, <32 x i32> poison, <32 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <32 x i32> [ %val1, %then ], [ %val2, %else ]
  ret <32 x i32> %val3
}

define <2 x bfloat> @shuffle_v2bf16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v2bf16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v2
; GFX9-NEXT:                                          ; implicit-def: $vgpr2
; GFX9-NEXT:    s_and_saveexec_b64 s[4:5], vcc
; GFX9-NEXT:    s_cbranch_execz .LBB15_2
; GFX9-NEXT:  ; %bb.1:                                ; %then
; GFX9-NEXT:    global_load_dword v0, v[0:1], off
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v2, v0, v0, s6
; GFX9-NEXT:  .LBB15_2:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v2bf16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX10-NEXT:                                          ; implicit-def: $vgpr2
; GFX10-NEXT:    s_and_saveexec_b32 s4, vcc_lo
; GFX10-NEXT:    s_cbranch_execz .LBB15_2
; GFX10-NEXT:  ; %bb.1:                                ; %then
; GFX10-NEXT:    global_load_dword v0, v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v2, v0, v0, 0x7060302
; GFX10-NEXT:  .LBB15_2:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v2bf16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX11-NEXT:                                          ; implicit-def: $vgpr2
; GFX11-NEXT:    s_and_saveexec_b32 s0, vcc_lo
; GFX11-NEXT:    s_cbranch_execz .LBB15_2
; GFX11-NEXT:  ; %bb.1:                                ; %then
; GFX11-NEXT:    global_load_b32 v0, v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v2, v0, v0, 0x7060302
; GFX11-NEXT:  .LBB15_2:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v0, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <2 x bfloat>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <2 x bfloat> %val0, <2 x bfloat> poison, <2 x i32> <i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <2 x bfloat> %val0, <2 x bfloat> poison, <2 x i32> <i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <2 x bfloat> [ %val1, %then ], [ %val2, %else ]
  ret <2 x bfloat> %val3
}

define <3 x bfloat> @shuffle_v3bf16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v3bf16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx2 v[0:1], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v2
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr2
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB16_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v2, v1, v1, s6
; GFX9-NEXT:  .LBB16_2:                               ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB16_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v2, v0, v0, s6
; GFX9-NEXT:    v_lshrrev_b32_e32 v1, 16, v0
; GFX9-NEXT:  .LBB16_4:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v0, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v3bf16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx2 v[0:1], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX10-NEXT:                                          ; implicit-def: $vgpr2
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB16_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v2, v1, v1, 0x5040100
; GFX10-NEXT:  .LBB16_2:                               ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB16_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v2, v0, v0, 0x7060302
; GFX10-NEXT:    v_lshrrev_b32_e32 v1, 16, v0
; GFX10-NEXT:  .LBB16_4:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v0, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v3bf16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b64 v[0:1], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX11-NEXT:                                          ; implicit-def: $vgpr2
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB16_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v2, v1, v1, 0x5040100
; GFX11-NEXT:  .LBB16_2:                               ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB16_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v2, v0, v0, 0x7060302
; GFX11-NEXT:    v_lshrrev_b32_e32 v1, 16, v0
; GFX11-NEXT:  .LBB16_4:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v0, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <3 x bfloat>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <3 x bfloat> %val0, <3 x bfloat> poison, <3 x i32> <i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <3 x bfloat> %val0, <3 x bfloat> poison, <3 x i32> <i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <3 x bfloat> [ %val1, %then ], [ %val2, %else ]
  ret <3 x bfloat> %val3
}

define <4 x bfloat> @shuffle_v4bf16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v4bf16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx2 v[3:4], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB17_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4
; GFX9-NEXT:  .LBB17_2:                               ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB17_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB17_4:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v4bf16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx2 v[3:4], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB17_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4
; GFX10-NEXT:  .LBB17_2:                               ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB17_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB17_4:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v4bf16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b64 v[3:4], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB17_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4
; GFX11-NEXT:  .LBB17_2:                               ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB17_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB17_4:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <4 x bfloat>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <4 x bfloat> %val0, <4 x bfloat> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <4 x bfloat> %val0, <4 x bfloat> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <4 x bfloat> [ %val1, %then ], [ %val2, %else ]
  ret <4 x bfloat> %val3
}

define <6 x bfloat> @shuffle_v6bf16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v6bf16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx3 v[3:5], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB18_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5
; GFX9-NEXT:  .LBB18_2:                               ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB18_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB18_4:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    v_mov_b32_e32 v2, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v6bf16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx3 v[3:5], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB18_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5
; GFX10-NEXT:  .LBB18_2:                               ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB18_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB18_4:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    v_mov_b32_e32 v2, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v6bf16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b96 v[3:5], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB18_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5
; GFX11-NEXT:  .LBB18_2:                               ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB18_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB18_4:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    v_mov_b32_e32 v2, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <6 x bfloat>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <6 x bfloat> %val0, <6 x bfloat> poison, <6 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <6 x bfloat> %val0, <6 x bfloat> poison, <6 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <6 x bfloat> [ %val1, %then ], [ %val2, %else ]
  ret <6 x bfloat> %val3
}

define <8 x bfloat> @shuffle_v8bf16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v8bf16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB19_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX9-NEXT:  .LBB19_2:                               ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB19_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB19_4:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    v_mov_b32_e32 v2, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v3, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v8bf16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB19_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX10-NEXT:  .LBB19_2:                               ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB19_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB19_4:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    v_mov_b32_e32 v2, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v3, v0
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v8bf16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b128 v[3:6], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB19_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX11-NEXT:  .LBB19_2:                               ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB19_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB19_4:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    v_mov_b32_e32 v2, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v3, v0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <8 x bfloat>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <8 x bfloat> %val0, <8 x bfloat> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <8 x bfloat> %val0, <8 x bfloat> poison, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <8 x bfloat> [ %val1, %then ], [ %val2, %else ]
  ret <8 x bfloat> %val3
}

define <16 x bfloat> @shuffle_v16bf16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v16bf16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB20_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX9-NEXT:  .LBB20_2:                               ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB20_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB20_4:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    v_mov_b32_e32 v2, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v3, v0
; GFX9-NEXT:    v_mov_b32_e32 v4, v0
; GFX9-NEXT:    v_mov_b32_e32 v5, v0
; GFX9-NEXT:    v_mov_b32_e32 v6, v0
; GFX9-NEXT:    v_mov_b32_e32 v7, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v16bf16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB20_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX10-NEXT:  .LBB20_2:                               ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB20_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB20_4:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    v_mov_b32_e32 v2, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v3, v0
; GFX10-NEXT:    v_mov_b32_e32 v4, v0
; GFX10-NEXT:    v_mov_b32_e32 v5, v0
; GFX10-NEXT:    v_mov_b32_e32 v6, v0
; GFX10-NEXT:    v_mov_b32_e32 v7, v0
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v16bf16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b128 v[3:6], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB20_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX11-NEXT:  .LBB20_2:                               ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB20_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB20_4:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    v_mov_b32_e32 v2, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v3, v0
; GFX11-NEXT:    v_mov_b32_e32 v4, v0
; GFX11-NEXT:    v_mov_b32_e32 v5, v0
; GFX11-NEXT:    v_mov_b32_e32 v6, v0
; GFX11-NEXT:    v_mov_b32_e32 v7, v0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <16 x bfloat>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <16 x bfloat> %val0, <16 x bfloat> poison, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <16 x bfloat> %val0, <16 x bfloat> poison, <16 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <16 x bfloat> [ %val1, %then ], [ %val2, %else ]
  ret <16 x bfloat> %val3
}

define <32 x bfloat> @shuffle_v32bf16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v32bf16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB21_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX9-NEXT:  .LBB21_2:                               ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB21_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB21_4:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    v_mov_b32_e32 v2, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v3, v0
; GFX9-NEXT:    v_mov_b32_e32 v4, v0
; GFX9-NEXT:    v_mov_b32_e32 v5, v0
; GFX9-NEXT:    v_mov_b32_e32 v6, v0
; GFX9-NEXT:    v_mov_b32_e32 v7, v0
; GFX9-NEXT:    v_mov_b32_e32 v8, v0
; GFX9-NEXT:    v_mov_b32_e32 v9, v0
; GFX9-NEXT:    v_mov_b32_e32 v10, v0
; GFX9-NEXT:    v_mov_b32_e32 v11, v0
; GFX9-NEXT:    v_mov_b32_e32 v12, v0
; GFX9-NEXT:    v_mov_b32_e32 v13, v0
; GFX9-NEXT:    v_mov_b32_e32 v14, v0
; GFX9-NEXT:    v_mov_b32_e32 v15, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v32bf16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB21_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX10-NEXT:  .LBB21_2:                               ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB21_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB21_4:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    v_mov_b32_e32 v2, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v3, v0
; GFX10-NEXT:    v_mov_b32_e32 v4, v0
; GFX10-NEXT:    v_mov_b32_e32 v5, v0
; GFX10-NEXT:    v_mov_b32_e32 v6, v0
; GFX10-NEXT:    v_mov_b32_e32 v7, v0
; GFX10-NEXT:    v_mov_b32_e32 v8, v0
; GFX10-NEXT:    v_mov_b32_e32 v9, v0
; GFX10-NEXT:    v_mov_b32_e32 v10, v0
; GFX10-NEXT:    v_mov_b32_e32 v11, v0
; GFX10-NEXT:    v_mov_b32_e32 v12, v0
; GFX10-NEXT:    v_mov_b32_e32 v13, v0
; GFX10-NEXT:    v_mov_b32_e32 v14, v0
; GFX10-NEXT:    v_mov_b32_e32 v15, v0
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v32bf16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b128 v[3:6], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB21_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX11-NEXT:  .LBB21_2:                               ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB21_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB21_4:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    v_mov_b32_e32 v2, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v3, v0
; GFX11-NEXT:    v_mov_b32_e32 v4, v0
; GFX11-NEXT:    v_mov_b32_e32 v5, v0
; GFX11-NEXT:    v_mov_b32_e32 v6, v0
; GFX11-NEXT:    v_mov_b32_e32 v7, v0
; GFX11-NEXT:    v_mov_b32_e32 v8, v0
; GFX11-NEXT:    v_mov_b32_e32 v9, v0
; GFX11-NEXT:    v_mov_b32_e32 v10, v0
; GFX11-NEXT:    v_mov_b32_e32 v11, v0
; GFX11-NEXT:    v_mov_b32_e32 v12, v0
; GFX11-NEXT:    v_mov_b32_e32 v13, v0
; GFX11-NEXT:    v_mov_b32_e32 v14, v0
; GFX11-NEXT:    v_mov_b32_e32 v15, v0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <32 x bfloat>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <32 x bfloat> %val0, <32 x bfloat> poison, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <32 x bfloat> %val0, <32 x bfloat> poison, <32 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <32 x bfloat> [ %val1, %then ], [ %val2, %else ]
  ret <32 x bfloat> %val3
}

define <2 x half> @shuffle_v2f16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v2f16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v2
; GFX9-NEXT:                                          ; implicit-def: $vgpr2
; GFX9-NEXT:    s_and_saveexec_b64 s[4:5], vcc
; GFX9-NEXT:    s_cbranch_execz .LBB22_2
; GFX9-NEXT:  ; %bb.1:                                ; %then
; GFX9-NEXT:    global_load_dword v0, v[0:1], off
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v2, v0, v0, s6
; GFX9-NEXT:  .LBB22_2:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v2f16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX10-NEXT:                                          ; implicit-def: $vgpr2
; GFX10-NEXT:    s_and_saveexec_b32 s4, vcc_lo
; GFX10-NEXT:    s_cbranch_execz .LBB22_2
; GFX10-NEXT:  ; %bb.1:                                ; %then
; GFX10-NEXT:    global_load_dword v0, v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v2, v0, v0, 0x7060302
; GFX10-NEXT:  .LBB22_2:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v2f16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX11-NEXT:                                          ; implicit-def: $vgpr2
; GFX11-NEXT:    s_and_saveexec_b32 s0, vcc_lo
; GFX11-NEXT:    s_cbranch_execz .LBB22_2
; GFX11-NEXT:  ; %bb.1:                                ; %then
; GFX11-NEXT:    global_load_b32 v0, v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v2, v0, v0, 0x7060302
; GFX11-NEXT:  .LBB22_2:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v0, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <2 x half>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <2 x half> %val0, <2 x half> poison, <2 x i32> <i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <2 x half> %val0, <2 x half> poison, <2 x i32> <i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <2 x half> [ %val1, %then ], [ %val2, %else ]
  ret <2 x half> %val3
}

define <3 x half> @shuffle_v3f16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v3f16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx2 v[0:1], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v2
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr2
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB23_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v2, v1, v1, s6
; GFX9-NEXT:  .LBB23_2:                               ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB23_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v2, v0, v0, s6
; GFX9-NEXT:    v_lshrrev_b32_e32 v1, 16, v0
; GFX9-NEXT:  .LBB23_4:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v0, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v3f16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx2 v[0:1], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX10-NEXT:                                          ; implicit-def: $vgpr2
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB23_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v2, v1, v1, 0x5040100
; GFX10-NEXT:  .LBB23_2:                               ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB23_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v2, v0, v0, 0x7060302
; GFX10-NEXT:    v_lshrrev_b32_e32 v1, 16, v0
; GFX10-NEXT:  .LBB23_4:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v0, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v3f16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b64 v[0:1], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v2, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v2
; GFX11-NEXT:                                          ; implicit-def: $vgpr2
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB23_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v2, v1, v1, 0x5040100
; GFX11-NEXT:  .LBB23_2:                               ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB23_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v2, v0, v0, 0x7060302
; GFX11-NEXT:    v_lshrrev_b32_e32 v1, 16, v0
; GFX11-NEXT:  .LBB23_4:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v0, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <3 x half>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <3 x half> %val0, <3 x half> poison, <3 x i32> <i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <3 x half> %val0, <3 x half> poison, <3 x i32> <i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <3 x half> [ %val1, %then ], [ %val2, %else ]
  ret <3 x half> %val3
}

define <4 x half> @shuffle_v4f16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v4f16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx2 v[3:4], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB24_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4
; GFX9-NEXT:  .LBB24_2:                               ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB24_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB24_4:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v4f16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx2 v[3:4], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB24_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4
; GFX10-NEXT:  .LBB24_2:                               ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB24_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB24_4:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v4f16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b64 v[3:4], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB24_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4
; GFX11-NEXT:  .LBB24_2:                               ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB24_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB24_4:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <4 x half>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <4 x half> %val0, <4 x half> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <4 x half> %val0, <4 x half> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <4 x half> [ %val1, %then ], [ %val2, %else ]
  ret <4 x half> %val3
}

define <6 x half> @shuffle_v6f16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v6f16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx3 v[3:5], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB25_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5
; GFX9-NEXT:  .LBB25_2:                               ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB25_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB25_4:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    v_mov_b32_e32 v2, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v6f16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx3 v[3:5], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB25_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5
; GFX10-NEXT:  .LBB25_2:                               ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB25_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB25_4:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    v_mov_b32_e32 v2, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v6f16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b96 v[3:5], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB25_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5
; GFX11-NEXT:  .LBB25_2:                               ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB25_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB25_4:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    v_mov_b32_e32 v2, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <6 x half>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <6 x half> %val0, <6 x half> poison, <6 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <6 x half> %val0, <6 x half> poison, <6 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <6 x half> [ %val1, %then ], [ %val2, %else ]
  ret <6 x half> %val3
}

define <8 x half> @shuffle_v8f16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v8f16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB26_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX9-NEXT:  .LBB26_2:                               ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB26_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB26_4:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    v_mov_b32_e32 v2, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v3, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v8f16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB26_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX10-NEXT:  .LBB26_2:                               ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB26_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB26_4:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    v_mov_b32_e32 v2, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v3, v0
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v8f16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b128 v[3:6], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB26_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX11-NEXT:  .LBB26_2:                               ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB26_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB26_4:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    v_mov_b32_e32 v2, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v3, v0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <8 x half>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <8 x half> %val0, <8 x half> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <8 x half> %val0, <8 x half> poison, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <8 x half> [ %val1, %then ], [ %val2, %else ]
  ret <8 x half> %val3
}

define <16 x half> @shuffle_v16f16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v16f16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB27_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX9-NEXT:  .LBB27_2:                               ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB27_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB27_4:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    v_mov_b32_e32 v2, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v3, v0
; GFX9-NEXT:    v_mov_b32_e32 v4, v0
; GFX9-NEXT:    v_mov_b32_e32 v5, v0
; GFX9-NEXT:    v_mov_b32_e32 v6, v0
; GFX9-NEXT:    v_mov_b32_e32 v7, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v16f16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB27_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX10-NEXT:  .LBB27_2:                               ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB27_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB27_4:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    v_mov_b32_e32 v2, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v3, v0
; GFX10-NEXT:    v_mov_b32_e32 v4, v0
; GFX10-NEXT:    v_mov_b32_e32 v5, v0
; GFX10-NEXT:    v_mov_b32_e32 v6, v0
; GFX10-NEXT:    v_mov_b32_e32 v7, v0
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v16f16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b128 v[3:6], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB27_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX11-NEXT:  .LBB27_2:                               ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB27_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB27_4:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    v_mov_b32_e32 v2, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v3, v0
; GFX11-NEXT:    v_mov_b32_e32 v4, v0
; GFX11-NEXT:    v_mov_b32_e32 v5, v0
; GFX11-NEXT:    v_mov_b32_e32 v6, v0
; GFX11-NEXT:    v_mov_b32_e32 v7, v0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <16 x half>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <16 x half> %val0, <16 x half> poison, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <16 x half> %val0, <16 x half> poison, <16 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <16 x half> [ %val1, %then ], [ %val2, %else ]
  ret <16 x half> %val3
}

define <32 x half> @shuffle_v32f16_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v32f16_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:                                          ; implicit-def: $vgpr0
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:    s_cbranch_execz .LBB28_2
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    s_mov_b32 s6, 0x5040100
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v4, v4, s6
; GFX9-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX9-NEXT:  .LBB28_2:                               ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:    s_cbranch_execz .LBB28_4
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    s_mov_b32 s6, 0x7060302
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_perm_b32 v0, v3, v3, s6
; GFX9-NEXT:  .LBB28_4:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v1, v0
; GFX9-NEXT:    v_mov_b32_e32 v2, v0
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v3, v0
; GFX9-NEXT:    v_mov_b32_e32 v4, v0
; GFX9-NEXT:    v_mov_b32_e32 v5, v0
; GFX9-NEXT:    v_mov_b32_e32 v6, v0
; GFX9-NEXT:    v_mov_b32_e32 v7, v0
; GFX9-NEXT:    v_mov_b32_e32 v8, v0
; GFX9-NEXT:    v_mov_b32_e32 v9, v0
; GFX9-NEXT:    v_mov_b32_e32 v10, v0
; GFX9-NEXT:    v_mov_b32_e32 v11, v0
; GFX9-NEXT:    v_mov_b32_e32 v12, v0
; GFX9-NEXT:    v_mov_b32_e32 v13, v0
; GFX9-NEXT:    v_mov_b32_e32 v14, v0
; GFX9-NEXT:    v_mov_b32_e32 v15, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v32f16_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_dwordx4 v[3:6], v[0:1], off
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:                                          ; implicit-def: $vgpr0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:    s_cbranch_execz .LBB28_2
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX10-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX10-NEXT:  .LBB28_2:                               ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:    s_cbranch_execz .LBB28_4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX10-NEXT:  .LBB28_4:                               ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v1, v0
; GFX10-NEXT:    v_mov_b32_e32 v2, v0
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v3, v0
; GFX10-NEXT:    v_mov_b32_e32 v4, v0
; GFX10-NEXT:    v_mov_b32_e32 v5, v0
; GFX10-NEXT:    v_mov_b32_e32 v6, v0
; GFX10-NEXT:    v_mov_b32_e32 v7, v0
; GFX10-NEXT:    v_mov_b32_e32 v8, v0
; GFX10-NEXT:    v_mov_b32_e32 v9, v0
; GFX10-NEXT:    v_mov_b32_e32 v10, v0
; GFX10-NEXT:    v_mov_b32_e32 v11, v0
; GFX10-NEXT:    v_mov_b32_e32 v12, v0
; GFX10-NEXT:    v_mov_b32_e32 v13, v0
; GFX10-NEXT:    v_mov_b32_e32 v14, v0
; GFX10-NEXT:    v_mov_b32_e32 v15, v0
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v32f16_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    global_load_b128 v[3:6], v[0:1], off
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:                                          ; implicit-def: $vgpr0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:    s_cbranch_execz .LBB28_2
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v4, v4, 0x5040100
; GFX11-NEXT:                                          ; implicit-def: $vgpr3_vgpr4_vgpr5_vgpr6
; GFX11-NEXT:  .LBB28_2:                               ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:    s_cbranch_execz .LBB28_4
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_perm_b32 v0, v3, v3, 0x7060302
; GFX11-NEXT:  .LBB28_4:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v1, v0
; GFX11-NEXT:    v_mov_b32_e32 v2, v0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v3, v0
; GFX11-NEXT:    v_mov_b32_e32 v4, v0
; GFX11-NEXT:    v_mov_b32_e32 v5, v0
; GFX11-NEXT:    v_mov_b32_e32 v6, v0
; GFX11-NEXT:    v_mov_b32_e32 v7, v0
; GFX11-NEXT:    v_mov_b32_e32 v8, v0
; GFX11-NEXT:    v_mov_b32_e32 v9, v0
; GFX11-NEXT:    v_mov_b32_e32 v10, v0
; GFX11-NEXT:    v_mov_b32_e32 v11, v0
; GFX11-NEXT:    v_mov_b32_e32 v12, v0
; GFX11-NEXT:    v_mov_b32_e32 v13, v0
; GFX11-NEXT:    v_mov_b32_e32 v14, v0
; GFX11-NEXT:    v_mov_b32_e32 v15, v0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <32 x half>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <32 x half> %val0, <32 x half> poison, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <32 x half> %val0, <32 x half> poison, <32 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <32 x half> [ %val1, %then ], [ %val2, %else ]
  ret <32 x half> %val3
}

define <2 x float> @shuffle_v2f32_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v2f32_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v3, v0
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX9-NEXT:    v_mov_b32_e32 v4, v1
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:                                          ; implicit-def: $vgpr1
; GFX9-NEXT:    s_and_saveexec_b64 s[4:5], vcc
; GFX9-NEXT:    s_cbranch_execz .LBB29_2
; GFX9-NEXT:  ; %bb.1:                                ; %then
; GFX9-NEXT:    global_load_dwordx2 v[0:1], v[3:4], off
; GFX9-NEXT:  .LBB29_2:                               ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v0, v1
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v2f32_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v3, v0
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX10-NEXT:    v_mov_b32_e32 v4, v1
; GFX10-NEXT:                                          ; implicit-def: $vgpr1
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:    s_and_saveexec_b32 s4, vcc_lo
; GFX10-NEXT:    s_cbranch_execz .LBB29_2
; GFX10-NEXT:  ; %bb.1:                                ; %then
; GFX10-NEXT:    global_load_dwordx2 v[0:1], v[3:4], off
; GFX10-NEXT:  .LBB29_2:                               ; %finally
; GFX10-NEXT:    s_waitcnt_depctr 0xffe3
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v0, v1
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v2f32_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v3, v0
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v2
; GFX11-NEXT:    s_mov_b32 s0, exec_lo
; GFX11-NEXT:                                          ; implicit-def: $vgpr1
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_cmpx_eq_u32_e32 1, v0
; GFX11-NEXT:    s_cbranch_execz .LBB29_2
; GFX11-NEXT:  ; %bb.1:                                ; %then
; GFX11-NEXT:    global_load_b64 v[0:1], v[3:4], off
; GFX11-NEXT:  .LBB29_2:                               ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v0, v1
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <2 x float>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <2 x float> %val0, <2 x float> poison, <2 x i32> <i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <2 x float> %val0, <2 x float> poison, <2 x i32> <i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <2 x float> [ %val1, %then ], [ %val2, %else ]
  ret <2 x float> %val3
}

define <3 x float> @shuffle_v3f32_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v3f32_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v3, v2
; GFX9-NEXT:    global_load_dwordx3 v[0:2], v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v3
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:  ; %bb.2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    v_mov_b32_e32 v2, v1
; GFX9-NEXT:  ; %bb.4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v1
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v3f32_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v3, v2
; GFX10-NEXT:    global_load_dwordx3 v[0:2], v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v3
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:  ; %bb.2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    v_mov_b32_e32 v2, v1
; GFX10-NEXT:  ; %bb.4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v1
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v3f32_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v3, v2
; GFX11-NEXT:    global_load_b96 v[0:2], v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v3
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    v_mov_b32_e32 v1, v2
; GFX11-NEXT:  ; %bb.2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v2, v1
; GFX11-NEXT:  ; %bb.4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_dual_mov_b32 v0, v1 :: v_dual_mov_b32 v1, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <3 x float>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <3 x float> %val0, <3 x float> poison, <3 x i32> <i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <3 x float> %val0, <3 x float> poison, <3 x i32> <i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <3 x float> [ %val1, %then ], [ %val2, %else ]
  ret <3 x float> %val3
}

define <4 x float> @shuffle_v4f32_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v4f32_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:  ; %bb.2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    v_mov_b32_e32 v2, v1
; GFX9-NEXT:  ; %bb.4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v1
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:    v_mov_b32_e32 v3, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v4f32_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:  ; %bb.2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    v_mov_b32_e32 v2, v1
; GFX10-NEXT:  ; %bb.4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v1
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:    v_mov_b32_e32 v3, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v4f32_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    v_mov_b32_e32 v1, v2
; GFX11-NEXT:  ; %bb.2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v2, v1
; GFX11-NEXT:  ; %bb.4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_dual_mov_b32 v0, v1 :: v_dual_mov_b32 v1, v2
; GFX11-NEXT:    v_mov_b32_e32 v3, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <4 x float>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <4 x float> %val0, <4 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <4 x float> %val0, <4 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <4 x float> [ %val1, %then ], [ %val2, %else ]
  ret <4 x float> %val3
}

define <6 x float> @shuffle_v6f32_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v6f32_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:  ; %bb.2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    v_mov_b32_e32 v2, v1
; GFX9-NEXT:  ; %bb.4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v1
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:    v_mov_b32_e32 v3, v2
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    v_mov_b32_e32 v5, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v6f32_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:  ; %bb.2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    v_mov_b32_e32 v2, v1
; GFX10-NEXT:  ; %bb.4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v1
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:    v_mov_b32_e32 v3, v2
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    v_mov_b32_e32 v5, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v6f32_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    v_mov_b32_e32 v1, v2
; GFX11-NEXT:  ; %bb.2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v2, v1
; GFX11-NEXT:  ; %bb.4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_dual_mov_b32 v0, v1 :: v_dual_mov_b32 v1, v2
; GFX11-NEXT:    v_mov_b32_e32 v3, v2
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    v_mov_b32_e32 v5, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <6 x float>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <6 x float> %val0, <6 x float> poison, <6 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <6 x float> %val0, <6 x float> poison, <6 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <6 x float> [ %val1, %then ], [ %val2, %else ]
  ret <6 x float> %val3
}

define <8 x float> @shuffle_v8f32_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v8f32_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:  ; %bb.2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    v_mov_b32_e32 v2, v1
; GFX9-NEXT:  ; %bb.4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v1
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:    v_mov_b32_e32 v3, v2
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    v_mov_b32_e32 v5, v2
; GFX9-NEXT:    v_mov_b32_e32 v6, v2
; GFX9-NEXT:    v_mov_b32_e32 v7, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v8f32_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:  ; %bb.2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    v_mov_b32_e32 v2, v1
; GFX10-NEXT:  ; %bb.4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v1
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:    v_mov_b32_e32 v3, v2
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    v_mov_b32_e32 v5, v2
; GFX10-NEXT:    v_mov_b32_e32 v6, v2
; GFX10-NEXT:    v_mov_b32_e32 v7, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v8f32_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    v_mov_b32_e32 v1, v2
; GFX11-NEXT:  ; %bb.2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v2, v1
; GFX11-NEXT:  ; %bb.4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_dual_mov_b32 v0, v1 :: v_dual_mov_b32 v1, v2
; GFX11-NEXT:    v_mov_b32_e32 v3, v2
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    v_mov_b32_e32 v5, v2
; GFX11-NEXT:    v_mov_b32_e32 v6, v2
; GFX11-NEXT:    v_mov_b32_e32 v7, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <8 x float>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <8 x float> %val0, <8 x float> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <8 x float> %val0, <8 x float> poison, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <8 x float> [ %val1, %then ], [ %val2, %else ]
  ret <8 x float> %val3
}

define <16 x float> @shuffle_v16f32_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v16f32_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:  ; %bb.2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    v_mov_b32_e32 v2, v1
; GFX9-NEXT:  ; %bb.4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v1
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:    v_mov_b32_e32 v3, v2
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    v_mov_b32_e32 v5, v2
; GFX9-NEXT:    v_mov_b32_e32 v6, v2
; GFX9-NEXT:    v_mov_b32_e32 v7, v2
; GFX9-NEXT:    v_mov_b32_e32 v8, v2
; GFX9-NEXT:    v_mov_b32_e32 v9, v2
; GFX9-NEXT:    v_mov_b32_e32 v10, v2
; GFX9-NEXT:    v_mov_b32_e32 v11, v2
; GFX9-NEXT:    v_mov_b32_e32 v12, v2
; GFX9-NEXT:    v_mov_b32_e32 v13, v2
; GFX9-NEXT:    v_mov_b32_e32 v14, v2
; GFX9-NEXT:    v_mov_b32_e32 v15, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v16f32_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:  ; %bb.2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    v_mov_b32_e32 v2, v1
; GFX10-NEXT:  ; %bb.4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v1
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:    v_mov_b32_e32 v3, v2
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    v_mov_b32_e32 v5, v2
; GFX10-NEXT:    v_mov_b32_e32 v6, v2
; GFX10-NEXT:    v_mov_b32_e32 v7, v2
; GFX10-NEXT:    v_mov_b32_e32 v8, v2
; GFX10-NEXT:    v_mov_b32_e32 v9, v2
; GFX10-NEXT:    v_mov_b32_e32 v10, v2
; GFX10-NEXT:    v_mov_b32_e32 v11, v2
; GFX10-NEXT:    v_mov_b32_e32 v12, v2
; GFX10-NEXT:    v_mov_b32_e32 v13, v2
; GFX10-NEXT:    v_mov_b32_e32 v14, v2
; GFX10-NEXT:    v_mov_b32_e32 v15, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v16f32_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    v_mov_b32_e32 v1, v2
; GFX11-NEXT:  ; %bb.2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v2, v1
; GFX11-NEXT:  ; %bb.4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_dual_mov_b32 v0, v1 :: v_dual_mov_b32 v1, v2
; GFX11-NEXT:    v_mov_b32_e32 v3, v2
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    v_mov_b32_e32 v5, v2
; GFX11-NEXT:    v_mov_b32_e32 v6, v2
; GFX11-NEXT:    v_mov_b32_e32 v7, v2
; GFX11-NEXT:    v_mov_b32_e32 v8, v2
; GFX11-NEXT:    v_mov_b32_e32 v9, v2
; GFX11-NEXT:    v_mov_b32_e32 v10, v2
; GFX11-NEXT:    v_mov_b32_e32 v11, v2
; GFX11-NEXT:    v_mov_b32_e32 v12, v2
; GFX11-NEXT:    v_mov_b32_e32 v13, v2
; GFX11-NEXT:    v_mov_b32_e32 v14, v2
; GFX11-NEXT:    v_mov_b32_e32 v15, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <16 x float>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <16 x float> %val0, <16 x float> poison, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <16 x float> %val0, <16 x float> poison, <16 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <16 x float> [ %val1, %then ], [ %val2, %else ]
  ret <16 x float> %val3
}

define <32 x float> @shuffle_v32f32_rebroadcast(ptr addrspace(1) %arg0, i1 %cond) {
; GFX9-LABEL: shuffle_v32f32_rebroadcast:
; GFX9:       ; %bb.0:                                ; %entry
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, 1, v0
; GFX9-NEXT:    s_xor_b64 s[4:5], vcc, -1
; GFX9-NEXT:    s_and_saveexec_b64 s[6:7], s[4:5]
; GFX9-NEXT:    s_xor_b64 s[4:5], exec, s[6:7]
; GFX9-NEXT:  ; %bb.1:                                ; %else
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:  ; %bb.2:                                ; %Flow
; GFX9-NEXT:    s_andn2_saveexec_b64 s[4:5], s[4:5]
; GFX9-NEXT:  ; %bb.3:                                ; %then
; GFX9-NEXT:    v_mov_b32_e32 v2, v1
; GFX9-NEXT:  ; %bb.4:                                ; %finally
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v1
; GFX9-NEXT:    v_mov_b32_e32 v1, v2
; GFX9-NEXT:    v_mov_b32_e32 v3, v2
; GFX9-NEXT:    v_mov_b32_e32 v4, v2
; GFX9-NEXT:    v_mov_b32_e32 v5, v2
; GFX9-NEXT:    v_mov_b32_e32 v6, v2
; GFX9-NEXT:    v_mov_b32_e32 v7, v2
; GFX9-NEXT:    v_mov_b32_e32 v8, v2
; GFX9-NEXT:    v_mov_b32_e32 v9, v2
; GFX9-NEXT:    v_mov_b32_e32 v10, v2
; GFX9-NEXT:    v_mov_b32_e32 v11, v2
; GFX9-NEXT:    v_mov_b32_e32 v12, v2
; GFX9-NEXT:    v_mov_b32_e32 v13, v2
; GFX9-NEXT:    v_mov_b32_e32 v14, v2
; GFX9-NEXT:    v_mov_b32_e32 v15, v2
; GFX9-NEXT:    v_mov_b32_e32 v16, v2
; GFX9-NEXT:    v_mov_b32_e32 v17, v2
; GFX9-NEXT:    v_mov_b32_e32 v18, v2
; GFX9-NEXT:    v_mov_b32_e32 v19, v2
; GFX9-NEXT:    v_mov_b32_e32 v20, v2
; GFX9-NEXT:    v_mov_b32_e32 v21, v2
; GFX9-NEXT:    v_mov_b32_e32 v22, v2
; GFX9-NEXT:    v_mov_b32_e32 v23, v2
; GFX9-NEXT:    v_mov_b32_e32 v24, v2
; GFX9-NEXT:    v_mov_b32_e32 v25, v2
; GFX9-NEXT:    v_mov_b32_e32 v26, v2
; GFX9-NEXT:    v_mov_b32_e32 v27, v2
; GFX9-NEXT:    v_mov_b32_e32 v28, v2
; GFX9-NEXT:    v_mov_b32_e32 v29, v2
; GFX9-NEXT:    v_mov_b32_e32 v30, v2
; GFX9-NEXT:    v_mov_b32_e32 v31, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v32f32_rebroadcast:
; GFX10:       ; %bb.0:                                ; %entry
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0)
; GFX10-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX10-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX10-NEXT:    s_xor_b32 s4, vcc_lo, -1
; GFX10-NEXT:    s_and_saveexec_b32 s5, s4
; GFX10-NEXT:    s_xor_b32 s4, exec_lo, s5
; GFX10-NEXT:  ; %bb.1:                                ; %else
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:  ; %bb.2:                                ; %Flow
; GFX10-NEXT:    s_andn2_saveexec_b32 s4, s4
; GFX10-NEXT:  ; %bb.3:                                ; %then
; GFX10-NEXT:    v_mov_b32_e32 v2, v1
; GFX10-NEXT:  ; %bb.4:                                ; %finally
; GFX10-NEXT:    s_or_b32 exec_lo, exec_lo, s4
; GFX10-NEXT:    v_mov_b32_e32 v0, v1
; GFX10-NEXT:    v_mov_b32_e32 v1, v2
; GFX10-NEXT:    v_mov_b32_e32 v3, v2
; GFX10-NEXT:    v_mov_b32_e32 v4, v2
; GFX10-NEXT:    v_mov_b32_e32 v5, v2
; GFX10-NEXT:    v_mov_b32_e32 v6, v2
; GFX10-NEXT:    v_mov_b32_e32 v7, v2
; GFX10-NEXT:    v_mov_b32_e32 v8, v2
; GFX10-NEXT:    v_mov_b32_e32 v9, v2
; GFX10-NEXT:    v_mov_b32_e32 v10, v2
; GFX10-NEXT:    v_mov_b32_e32 v11, v2
; GFX10-NEXT:    v_mov_b32_e32 v12, v2
; GFX10-NEXT:    v_mov_b32_e32 v13, v2
; GFX10-NEXT:    v_mov_b32_e32 v14, v2
; GFX10-NEXT:    v_mov_b32_e32 v15, v2
; GFX10-NEXT:    v_mov_b32_e32 v16, v2
; GFX10-NEXT:    v_mov_b32_e32 v17, v2
; GFX10-NEXT:    v_mov_b32_e32 v18, v2
; GFX10-NEXT:    v_mov_b32_e32 v19, v2
; GFX10-NEXT:    v_mov_b32_e32 v20, v2
; GFX10-NEXT:    v_mov_b32_e32 v21, v2
; GFX10-NEXT:    v_mov_b32_e32 v22, v2
; GFX10-NEXT:    v_mov_b32_e32 v23, v2
; GFX10-NEXT:    v_mov_b32_e32 v24, v2
; GFX10-NEXT:    v_mov_b32_e32 v25, v2
; GFX10-NEXT:    v_mov_b32_e32 v26, v2
; GFX10-NEXT:    v_mov_b32_e32 v27, v2
; GFX10-NEXT:    v_mov_b32_e32 v28, v2
; GFX10-NEXT:    v_mov_b32_e32 v29, v2
; GFX10-NEXT:    v_mov_b32_e32 v30, v2
; GFX10-NEXT:    v_mov_b32_e32 v31, v2
; GFX10-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v32f32_rebroadcast:
; GFX11:       ; %bb.0:                                ; %entry
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:    s_waitcnt vmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v4
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
; GFX11-NEXT:    v_cmp_eq_u32_e32 vcc_lo, 1, v0
; GFX11-NEXT:    s_xor_b32 s0, vcc_lo, -1
; GFX11-NEXT:    s_and_saveexec_b32 s1, s0
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; GFX11-NEXT:    s_xor_b32 s0, exec_lo, s1
; GFX11-NEXT:  ; %bb.1:                                ; %else
; GFX11-NEXT:    v_mov_b32_e32 v1, v2
; GFX11-NEXT:  ; %bb.2:                                ; %Flow
; GFX11-NEXT:    s_and_not1_saveexec_b32 s0, s0
; GFX11-NEXT:  ; %bb.3:                                ; %then
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_mov_b32_e32 v2, v1
; GFX11-NEXT:  ; %bb.4:                                ; %finally
; GFX11-NEXT:    s_or_b32 exec_lo, exec_lo, s0
; GFX11-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:    v_dual_mov_b32 v0, v1 :: v_dual_mov_b32 v1, v2
; GFX11-NEXT:    v_mov_b32_e32 v3, v2
; GFX11-NEXT:    v_mov_b32_e32 v4, v2
; GFX11-NEXT:    v_mov_b32_e32 v5, v2
; GFX11-NEXT:    v_mov_b32_e32 v6, v2
; GFX11-NEXT:    v_mov_b32_e32 v7, v2
; GFX11-NEXT:    v_mov_b32_e32 v8, v2
; GFX11-NEXT:    v_mov_b32_e32 v9, v2
; GFX11-NEXT:    v_mov_b32_e32 v10, v2
; GFX11-NEXT:    v_mov_b32_e32 v11, v2
; GFX11-NEXT:    v_mov_b32_e32 v12, v2
; GFX11-NEXT:    v_mov_b32_e32 v13, v2
; GFX11-NEXT:    v_mov_b32_e32 v14, v2
; GFX11-NEXT:    v_mov_b32_e32 v15, v2
; GFX11-NEXT:    v_mov_b32_e32 v16, v2
; GFX11-NEXT:    v_mov_b32_e32 v17, v2
; GFX11-NEXT:    v_mov_b32_e32 v18, v2
; GFX11-NEXT:    v_mov_b32_e32 v19, v2
; GFX11-NEXT:    v_mov_b32_e32 v20, v2
; GFX11-NEXT:    v_mov_b32_e32 v21, v2
; GFX11-NEXT:    v_mov_b32_e32 v22, v2
; GFX11-NEXT:    v_mov_b32_e32 v23, v2
; GFX11-NEXT:    v_mov_b32_e32 v24, v2
; GFX11-NEXT:    v_mov_b32_e32 v25, v2
; GFX11-NEXT:    v_mov_b32_e32 v26, v2
; GFX11-NEXT:    v_mov_b32_e32 v27, v2
; GFX11-NEXT:    v_mov_b32_e32 v28, v2
; GFX11-NEXT:    v_mov_b32_e32 v29, v2
; GFX11-NEXT:    v_mov_b32_e32 v30, v2
; GFX11-NEXT:    v_mov_b32_e32 v31, v2
; GFX11-NEXT:    s_setpc_b64 s[30:31]
entry:
  %val0 = load <32 x float>, ptr addrspace(1) %arg0
  br i1 %cond, label %then, label %else

then:
  %val1 = shufflevector <32 x float> %val0, <32 x float> poison, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  br label %finally

else:
  %val2 = shufflevector <32 x float> %val0, <32 x float> poison, <32 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  br label %finally

finally:
  %val3 = phi <32 x float> [ %val1, %then ], [ %val2, %else ]
  ret <32 x float> %val3
}
