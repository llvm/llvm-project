; RUN:  llc -global-isel=0 -march=amdgcn -mcpu=gfx704 -verify-machineinstrs < %s  | FileCheck --check-prefixes=GFX7SELDAG %s
; RUN:  llc -global-isel=0 -march=amdgcn -mcpu=gfx803 -verify-machineinstrs < %s  | FileCheck --check-prefixes=GFX8SELDAG,GFX8CHECK %s
; RUN:  llc -global-isel=1 -march=amdgcn -mcpu=gfx803 -verify-machineinstrs < %s  | FileCheck --check-prefixes=GFX8GLISEL,GFX8CHECK %s
; RUN:  llc -global-isel=0 -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s  | FileCheck --check-prefixes=GFX9SELDAG,GFX9CHECK %s
; RUN:  llc -global-isel=1 -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s  | FileCheck --check-prefixes=GFX9GLISEL,GFX9CHECK %s
; RUN:  llc -global-isel=0 -march=amdgcn -mcpu=gfx1031 -verify-machineinstrs < %s | FileCheck --check-prefixes=GFX10SELDAG,GFX10CHECK %s
; RUN:  llc -global-isel=1 -march=amdgcn -mcpu=gfx1031 -verify-machineinstrs < %s | FileCheck --check-prefixes=GFX10GLISEL,GFX10CHECK %s
; RUN:  llc -global-isel=0 -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck --check-prefixes=GFX11SELDAG,GFX11CHECK %s
; RUN:  llc -global-isel=1 -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck --check-prefixes=GFX11GLISEL,GFX11CHECK %s

; GFX7SELDAG-LABEL: sgpr_isnan_f16:
; GFX7SELDAG:       ; %bb.0:
; GFX7SELDAG-NEXT:    s_load_dword s4, s[0:1], 0xb
; GFX7SELDAG-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; GFX7SELDAG-NEXT:    s_mov_b32 s3, 0xf000
; GFX7SELDAG-NEXT:    s_mov_b32 s2, -1
; GFX7SELDAG-NEXT:    s_waitcnt lgkmcnt(0)
; GFX7SELDAG-NEXT:    s_and_b32 s4, s4, 0x7fff
; GFX7SELDAG-NEXT:    s_cmpk_gt_i32 s4, 0x7c00
; GFX7SELDAG-NEXT:    s_cselect_b64 s[4:5], -1, 0
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, -1, s[4:5]
; GFX7SELDAG-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GFX7SELDAG-NEXT:    s_endpgm
;
; GFX8CHECK-LABEL: sgpr_isnan_f16:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_load_dword s2, s[0:1], 0x2c
; GFX8CHECK-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; GFX8CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f16_e64 s[2:3], s2, 3
; GFX8CHECK-NEXT:    v_mov_b32_e32 v0, s0
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, -1, s[2:3]
; GFX8CHECK-NEXT:    v_mov_b32_e32 v1, s1
; GFX8CHECK-NEXT:    flat_store_dword v[0:1], v2
; GFX8CHECK-NEXT:    s_endpgm
;
; GFX9CHECK-LABEL: sgpr_isnan_f16:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_load_dword s4, s[0:1], 0x2c
; GFX9CHECK-NEXT:    s_load_dwordx2 s[2:3], s[0:1], 0x24
; GFX9CHECK-NEXT:    v_mov_b32_e32 v0, 0
; GFX9CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f16_e64 s[0:1], s4, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, -1, s[0:1]
; GFX9CHECK-NEXT:    global_store_dword v0, v1, s[2:3]
; GFX9CHECK-NEXT:    s_endpgm
;
; GFX10CHECK-LABEL: sgpr_isnan_f16:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_clause 0x1
; GFX10CHECK-NEXT:    s_load_dword s2, s[0:1], 0x2c
; GFX10CHECK-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; GFX10CHECK-NEXT:    v_mov_b32_e32 v0, 0
; GFX10CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; GFX10CHECK-NEXT:    v_cmp_class_f16_e64 s2, s2, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, -1, s2
; GFX10CHECK-NEXT:    global_store_dword v0, v1, s[0:1]
; GFX10CHECK-NEXT:    s_endpgm
;
; GFX11CHECK-LABEL: sgpr_isnan_f16:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_clause 0x1
; GFX11CHECK-NEXT:    s_load_b32 s2, s[0:1], 0x2c
; GFX11CHECK-NEXT:    s_load_b64 s[0:1], s[0:1], 0x24
; GFX11CHECK-NEXT:    v_mov_b32_e32 v0, 0
; GFX11CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11CHECK-NEXT:    v_cmp_class_f16_e64 s2, s2, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, -1, s2
; GFX11CHECK-NEXT:    global_store_b32 v0, v1, s[0:1]
; GFX11CHECK-NEXT:    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
; GFX11CHECK-NEXT:    s_endpgm
define amdgpu_kernel void @sgpr_isnan_f16(i32 addrspace(1)* %out, half %x) {
  %result = call i1 @llvm.is.fpclass.f16(half %x, i32 3)  ; nan
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

define i1 @isnan_f16(half %x) nounwind {
; GFX7SELDAG-LABEL: isnan_f16:
; GFX7SELDAG:       ; %bb.0:
; GFX7SELDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7SELDAG-NEXT:    v_cvt_f16_f32_e32 v0, v0
; GFX7SELDAG-NEXT:    s_movk_i32 s4, 0x7c00
; GFX7SELDAG-NEXT:    v_and_b32_e32 v0, 0x7fff, v0
; GFX7SELDAG-NEXT:    v_cmp_lt_i32_e32 vcc, s4, v0
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7SELDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_f16:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f16_e64 s[4:5], v0, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_f16:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f16_e64 s[4:5], v0, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_f16:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f16_e64 s4, v0, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_f16:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f16_e64 s0, v0, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f16(half %x, i32 3)  ; nan
  ret i1 %1
}

define <2 x i1> @isnan_v2f16(<2 x half> %x) nounwind {
; GFX7SELDAG-LABEL: isnan_v2f16:
; GFX7SELDAG:       ; %bb.0:
; GFX7SELDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7SELDAG-NEXT:    v_cvt_f16_f32_e32 v0, v0
; GFX7SELDAG-NEXT:    v_cvt_f16_f32_e32 v1, v1
; GFX7SELDAG-NEXT:    s_movk_i32 s4, 0x7c00
; GFX7SELDAG-NEXT:    v_and_b32_e32 v0, 0x7fff, v0
; GFX7SELDAG-NEXT:    v_and_b32_e32 v1, 0x7fff, v1
; GFX7SELDAG-NEXT:    v_cmp_lt_i32_e32 vcc, s4, v0
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7SELDAG-NEXT:    v_cmp_lt_i32_e32 vcc, s4, v1
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v1, 0, 1, vcc
; GFX7SELDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_v2f16:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_lshrrev_b32_e32 v1, 16, v0
; GFX8CHECK-NEXT:    v_cmp_class_f16_e64 s[4:5], v0, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f16_e64 s[4:5], v1, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_v2f16:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_mov_b32_e32 v1, 3
; GFX9CHECK-NEXT:    v_cmp_class_f16_e64 s[4:5], v0, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f16_sdwa s[4:5], v0, v1 src0_sel:WORD_1 src1_sel:DWORD
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_mov_b32_e32 v0, v2
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_v2f16:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_mov_b32_e32 v1, 3
; GFX10CHECK-NEXT:    v_cmp_class_f16_e64 s4, v0, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f16_sdwa s4, v0, v1 src0_sel:WORD_1 src1_sel:DWORD
; GFX10CHECK-NEXT:    v_mov_b32_e32 v0, v2
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_v2f16:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_lshrrev_b32_e32 v1, 16, v0
; GFX11CHECK-NEXT:    v_cmp_class_f16_e64 s0, v0, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f16_e64 s0, v1, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call <2 x i1> @llvm.is.fpclass.v2f16(<2 x half> %x, i32 3)  ; nan
  ret <2 x i1> %1
}

define <3 x i1> @isnan_v3f16(<3 x half> %x) nounwind {
; GFX7SELDAG-LABEL: isnan_v3f16:
; GFX7SELDAG:       ; %bb.0:
; GFX7SELDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7SELDAG-NEXT:    v_cvt_f16_f32_e32 v0, v0
; GFX7SELDAG-NEXT:    v_cvt_f16_f32_e32 v1, v1
; GFX7SELDAG-NEXT:    v_cvt_f16_f32_e32 v2, v2
; GFX7SELDAG-NEXT:    s_movk_i32 s4, 0x7c00
; GFX7SELDAG-NEXT:    v_and_b32_e32 v0, 0x7fff, v0
; GFX7SELDAG-NEXT:    v_and_b32_e32 v1, 0x7fff, v1
; GFX7SELDAG-NEXT:    v_cmp_lt_i32_e32 vcc, s4, v0
; GFX7SELDAG-NEXT:    v_and_b32_e32 v2, 0x7fff, v2
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7SELDAG-NEXT:    v_cmp_lt_i32_e32 vcc, s4, v1
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v1, 0, 1, vcc
; GFX7SELDAG-NEXT:    v_cmp_lt_i32_e32 vcc, s4, v2
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v2, 0, 1, vcc
; GFX7SELDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8SELDAG-LABEL: isnan_v3f16:
; GFX8SELDAG:       ; %bb.0:
; GFX8SELDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8SELDAG-NEXT:    v_lshrrev_b32_e32 v2, 16, v0
; GFX8SELDAG-NEXT:    v_cmp_u_f16_e32 vcc, v2, v2
; GFX8SELDAG-NEXT:    v_cndmask_b32_e64 v3, 0, 1, vcc
; GFX8SELDAG-NEXT:    v_cmp_u_f16_e32 vcc, v0, v0
; GFX8SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX8SELDAG-NEXT:    v_cmp_u_f16_e32 vcc, v1, v1
; GFX8SELDAG-NEXT:    v_cndmask_b32_e64 v2, 0, 1, vcc
; GFX8SELDAG-NEXT:    v_mov_b32_e32 v1, v3
; GFX8SELDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8GLISEL-LABEL: isnan_v3f16:
; GFX8GLISEL:       ; %bb.0:
; GFX8GLISEL-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8GLISEL-NEXT:    v_lshrrev_b32_e32 v2, 16, v0
; GFX8GLISEL-NEXT:    v_cmp_class_f16_e64 s[4:5], v0, 3
; GFX8GLISEL-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8GLISEL-NEXT:    v_cmp_class_f16_e64 s[4:5], v2, 3
; GFX8GLISEL-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX8GLISEL-NEXT:    v_cmp_class_f16_e64 s[4:5], v1, 3
; GFX8GLISEL-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX8GLISEL-NEXT:    v_mov_b32_e32 v1, v3
; GFX8GLISEL-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9SELDAG-LABEL: isnan_v3f16:
; GFX9SELDAG:       ; %bb.0:
; GFX9SELDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9SELDAG-NEXT:    v_cmp_u_f16_sdwa s[4:5], v0, v0 src0_sel:WORD_1 src1_sel:WORD_1
; GFX9SELDAG-NEXT:    v_cmp_u_f16_e32 vcc, v0, v0
; GFX9SELDAG-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX9SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX9SELDAG-NEXT:    v_cmp_u_f16_e32 vcc, v1, v1
; GFX9SELDAG-NEXT:    v_cndmask_b32_e64 v2, 0, 1, vcc
; GFX9SELDAG-NEXT:    v_mov_b32_e32 v1, v3
; GFX9SELDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9GLISEL-LABEL: isnan_v3f16:
; GFX9GLISEL:       ; %bb.0:
; GFX9GLISEL-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9GLISEL-NEXT:    v_mov_b32_e32 v2, 3
; GFX9GLISEL-NEXT:    v_cmp_class_f16_e64 s[4:5], v0, 3
; GFX9GLISEL-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX9GLISEL-NEXT:    v_cmp_class_f16_sdwa s[4:5], v0, v2 src0_sel:WORD_1 src1_sel:DWORD
; GFX9GLISEL-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX9GLISEL-NEXT:    v_cmp_class_f16_e64 s[4:5], v1, 3
; GFX9GLISEL-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX9GLISEL-NEXT:    v_mov_b32_e32 v0, v4
; GFX9GLISEL-NEXT:    v_mov_b32_e32 v1, v3
; GFX9GLISEL-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10SELDAG-LABEL: isnan_v3f16:
; GFX10SELDAG:       ; %bb.0:
; GFX10SELDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10SELDAG-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10SELDAG-NEXT:    v_cmp_u_f16_sdwa s4, v0, v0 src0_sel:WORD_1 src1_sel:WORD_1
; GFX10SELDAG-NEXT:    v_cmp_u_f16_e32 vcc_lo, v0, v0
; GFX10SELDAG-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s4
; GFX10SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc_lo
; GFX10SELDAG-NEXT:    v_cmp_u_f16_e32 vcc_lo, v1, v1
; GFX10SELDAG-NEXT:    v_mov_b32_e32 v1, v3
; GFX10SELDAG-NEXT:    v_cndmask_b32_e64 v2, 0, 1, vcc_lo
; GFX10SELDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10GLISEL-LABEL: isnan_v3f16:
; GFX10GLISEL:       ; %bb.0:
; GFX10GLISEL-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10GLISEL-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10GLISEL-NEXT:    v_mov_b32_e32 v2, 3
; GFX10GLISEL-NEXT:    v_cmp_class_f16_e64 s4, v0, 3
; GFX10GLISEL-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s4
; GFX10GLISEL-NEXT:    v_cmp_class_f16_sdwa s4, v0, v2 src0_sel:WORD_1 src1_sel:DWORD
; GFX10GLISEL-NEXT:    v_mov_b32_e32 v0, v4
; GFX10GLISEL-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s4
; GFX10GLISEL-NEXT:    v_cmp_class_f16_e64 s4, v1, 3
; GFX10GLISEL-NEXT:    v_mov_b32_e32 v1, v3
; GFX10GLISEL-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s4
; GFX10GLISEL-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11SELDAG-LABEL: isnan_v3f16:
; GFX11SELDAG:       ; %bb.0:
; GFX11SELDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11SELDAG-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11SELDAG-NEXT:    v_lshrrev_b32_e32 v2, 16, v0
; GFX11SELDAG-NEXT:    v_cmp_u_f16_e32 vcc_lo, v0, v0
; GFX11SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc_lo
; GFX11SELDAG-NEXT:    s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
; GFX11SELDAG-NEXT:    v_cmp_u_f16_e32 vcc_lo, v2, v2
; GFX11SELDAG-NEXT:    v_cndmask_b32_e64 v3, 0, 1, vcc_lo
; GFX11SELDAG-NEXT:    v_cmp_u_f16_e32 vcc_lo, v1, v1
; GFX11SELDAG-NEXT:    v_mov_b32_e32 v1, v3
; GFX11SELDAG-NEXT:    v_cndmask_b32_e64 v2, 0, 1, vcc_lo
; GFX11SELDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11GLISEL-LABEL: isnan_v3f16:
; GFX11GLISEL:       ; %bb.0:
; GFX11GLISEL-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11GLISEL-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11GLISEL-NEXT:    v_lshrrev_b32_e32 v2, 16, v0
; GFX11GLISEL-NEXT:    v_cmp_class_f16_e64 s0, v0, 3
; GFX11GLISEL-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
; GFX11GLISEL-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11GLISEL-NEXT:    v_cmp_class_f16_e64 s0, v2, 3
; GFX11GLISEL-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
; GFX11GLISEL-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s0
; GFX11GLISEL-NEXT:    v_cmp_class_f16_e64 s0, v1, 3
; GFX11GLISEL-NEXT:    v_mov_b32_e32 v1, v3
; GFX11GLISEL-NEXT:    s_delay_alu instid0(VALU_DEP_2)
; GFX11GLISEL-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11GLISEL-NEXT:    s_setpc_b64 s[30:31]
  %1 = call <3 x i1> @llvm.is.fpclass.v3f16(<3 x half> %x, i32 3)  ; nan
  ret <3 x i1> %1
}

define <4 x i1> @isnan_v4f16(<4 x half> %x) nounwind {
; GFX7SELDAG-LABEL: isnan_v4f16:
; GFX7SELDAG:       ; %bb.0:
; GFX7SELDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7SELDAG-NEXT:    v_cvt_f16_f32_e32 v0, v0
; GFX7SELDAG-NEXT:    v_cvt_f16_f32_e32 v1, v1
; GFX7SELDAG-NEXT:    v_cvt_f16_f32_e32 v2, v2
; GFX7SELDAG-NEXT:    v_cvt_f16_f32_e32 v3, v3
; GFX7SELDAG-NEXT:    s_movk_i32 s4, 0x7c00
; GFX7SELDAG-NEXT:    v_and_b32_e32 v0, 0x7fff, v0
; GFX7SELDAG-NEXT:    v_and_b32_e32 v1, 0x7fff, v1
; GFX7SELDAG-NEXT:    v_cmp_lt_i32_e32 vcc, s4, v0
; GFX7SELDAG-NEXT:    v_and_b32_e32 v2, 0x7fff, v2
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7SELDAG-NEXT:    v_cmp_lt_i32_e32 vcc, s4, v1
; GFX7SELDAG-NEXT:    v_and_b32_e32 v3, 0x7fff, v3
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v1, 0, 1, vcc
; GFX7SELDAG-NEXT:    v_cmp_lt_i32_e32 vcc, s4, v2
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v2, 0, 1, vcc
; GFX7SELDAG-NEXT:    v_cmp_lt_i32_e32 vcc, s4, v3
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v3, 0, 1, vcc
; GFX7SELDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8SELDAG-LABEL: isnan_v4f16:
; GFX8SELDAG:       ; %bb.0:
; GFX8SELDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8SELDAG-NEXT:    v_cmp_class_f16_e64 s[4:5], v0, 3
; GFX8SELDAG-NEXT:    v_lshrrev_b32_e32 v3, 16, v0
; GFX8SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8SELDAG-NEXT:    v_cmp_class_f16_e64 s[4:5], v1, 3
; GFX8SELDAG-NEXT:    v_lshrrev_b32_e32 v4, 16, v1
; GFX8SELDAG-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX8SELDAG-NEXT:    v_cmp_class_f16_e64 s[4:5], v3, 3
; GFX8SELDAG-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX8SELDAG-NEXT:    v_cmp_class_f16_e64 s[4:5], v4, 3
; GFX8SELDAG-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX8SELDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8GLISEL-LABEL: isnan_v4f16:
; GFX8GLISEL:       ; %bb.0:
; GFX8GLISEL-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8GLISEL-NEXT:    v_lshrrev_b32_e32 v2, 16, v0
; GFX8GLISEL-NEXT:    v_cmp_class_f16_e64 s[4:5], v0, 3
; GFX8GLISEL-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8GLISEL-NEXT:    v_cmp_class_f16_e64 s[4:5], v2, 3
; GFX8GLISEL-NEXT:    v_lshrrev_b32_e32 v3, 16, v1
; GFX8GLISEL-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX8GLISEL-NEXT:    v_cmp_class_f16_e64 s[4:5], v1, 3
; GFX8GLISEL-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX8GLISEL-NEXT:    v_cmp_class_f16_e64 s[4:5], v3, 3
; GFX8GLISEL-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX8GLISEL-NEXT:    v_mov_b32_e32 v1, v4
; GFX8GLISEL-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9SELDAG-LABEL: isnan_v4f16:
; GFX9SELDAG:       ; %bb.0:
; GFX9SELDAG-NEXT:   s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9SELDAG-NEXT:   v_cmp_class_f16_e64 s[4:5], v0, 3
; GFX9SELDAG-NEXT:   v_mov_b32_e32 v3, 3
; GFX9SELDAG-NEXT:   v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX9SELDAG-NEXT:   v_cmp_class_f16_e64 s[4:5], v1, 3
; GFX9SELDAG-NEXT:   v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX9SELDAG-NEXT:   v_cmp_class_f16_sdwa s[4:5], v0, v3 src0_sel:WORD_1 src1_sel:DWORD
; GFX9SELDAG-NEXT:   v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX9SELDAG-NEXT:   v_cmp_class_f16_sdwa s[4:5], v1, v3 src0_sel:WORD_1 src1_sel:DWORD
; GFX9SELDAG-NEXT:   v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX9SELDAG-NEXT:   v_mov_b32_e32 v0, v5
; GFX9SELDAG-NEXT:   v_mov_b32_e32 v1, v4
; GFX9SELDAG-NEXT:   s_setpc_b64 s[30:31]
;
; GFX9GLISEL-LABEL: isnan_v4f16:
; GFX9GLISEL:       ; %bb.0:
; GFX9GLISEL-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9GLISEL-NEXT:    v_mov_b32_e32 v3, 3
; GFX9GLISEL-NEXT:    v_cmp_class_f16_e64 s[4:5], v0, 3
; GFX9GLISEL-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX9GLISEL-NEXT:    v_cmp_class_f16_sdwa s[4:5], v0, v3 src0_sel:WORD_1 src1_sel:DWORD
; GFX9GLISEL-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX9GLISEL-NEXT:    v_cmp_class_f16_e64 s[4:5], v1, 3
; GFX9GLISEL-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX9GLISEL-NEXT:    v_cmp_class_f16_sdwa s[4:5], v1, v3 src0_sel:WORD_1 src1_sel:DWORD
; GFX9GLISEL-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX9GLISEL-NEXT:    v_mov_b32_e32 v0, v4
; GFX9GLISEL-NEXT:    v_mov_b32_e32 v1, v5
; GFX9GLISEL-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10SELDAG-LABEL: isnan_v4f16:
; GFX10SELDAG:       ; %bb.0:
; GFX10SELDAG-NEXT:   s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10SELDAG-NEXT:   s_waitcnt_vscnt null, 0x0
; GFX10SELDAG-NEXT:   v_mov_b32_e32 v2, 3
; GFX10SELDAG-NEXT:   v_cmp_class_f16_e64 s5, v0, 3
; GFX10SELDAG-NEXT:   v_cmp_class_f16_sdwa s4, v1, v2 src0_sel:WORD_1 src1_sel:DWORD
; GFX10SELDAG-NEXT:   v_cndmask_b32_e64 v4, 0, 1, s5
; GFX10SELDAG-NEXT:   v_cmp_class_f16_sdwa s5, v0, v2 src0_sel:WORD_1 src1_sel:DWORD
; GFX10SELDAG-NEXT:   v_cndmask_b32_e64 v3, 0, 1, s4
; GFX10SELDAG-NEXT:   v_mov_b32_e32 v0, v4
; GFX10SELDAG-NEXT:   v_cndmask_b32_e64 v5, 0, 1, s5
; GFX10SELDAG-NEXT:   v_cmp_class_f16_e64 s5, v1, 3
; GFX10SELDAG-NEXT:   v_mov_b32_e32 v1, v5
; GFX10SELDAG-NEXT:   v_cndmask_b32_e64 v2, 0, 1, s5
; GFX10SELDAG-NEXT:   s_setpc_b64 s[30:31]
;
; GFX10GLISEL-LABEL: isnan_v4f16:
; GFX10GLISEL:       ; %bb.0:
; GFX10GLISEL-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10GLISEL-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10GLISEL-NEXT:    v_mov_b32_e32 v3, 3
; GFX10GLISEL-NEXT:    v_cmp_class_f16_e64 s4, v0, 3
; GFX10GLISEL-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s4
; GFX10GLISEL-NEXT:    v_cmp_class_f16_sdwa s4, v0, v3 src0_sel:WORD_1 src1_sel:DWORD
; GFX10GLISEL-NEXT:    v_mov_b32_e32 v0, v4
; GFX10GLISEL-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s4
; GFX10GLISEL-NEXT:    v_cmp_class_f16_e64 s4, v1, 3
; GFX10GLISEL-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s4
; GFX10GLISEL-NEXT:    v_cmp_class_f16_sdwa s4, v1, v3 src0_sel:WORD_1 src1_sel:DWORD
; GFX10GLISEL-NEXT:    v_mov_b32_e32 v1, v5
; GFX10GLISEL-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s4
; GFX10GLISEL-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_v4f16:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f16_e64 s0, v0, 3
; GFX11CHECK-NEXT:    v_lshrrev_b32_e32 v3, 16, v0
; GFX11CHECK-NEXT:    v_lshrrev_b32_e32 v4, 16, v1
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f16_e64 s0, v1, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f16_e64 s0, v3, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f16_e64 s0, v4, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call <4 x i1> @llvm.is.fpclass.v4f16(<4 x half> %x, i32 3)  ; nan
  ret <4 x i1> %1
}

define i1 @isnan_f16_strictfp(half %x) strictfp nounwind {
; GFX7SELDAG-LABEL: isnan_f16_strictfp:
; GFX7SELDAG:       ; %bb.0:
; GFX7SELDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7SELDAG-NEXT:    v_cvt_f16_f32_e32 v0, v0
; GFX7SELDAG-NEXT:    s_movk_i32 s4, 0x7c00
; GFX7SELDAG-NEXT:    v_and_b32_e32 v0, 0x7fff, v0
; GFX7SELDAG-NEXT:    v_cmp_lt_i32_e32 vcc, s4, v0
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7SELDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_f16_strictfp:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f16_e64 s[4:5], v0, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_f16_strictfp:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f16_e64 s[4:5], v0, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_f16_strictfp:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f16_e64 s4, v0, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_f16_strictfp:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f16_e64 s0, v0, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f16(half %x, i32 3)  ; nan
  ret i1 %1
}

define i1 @isinf_f16(half %x) nounwind {
; GFX7SELDAG-LABEL: isinf_f16:
; GFX7SELDAG:       ; %bb.0:
; GFX7SELDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7SELDAG-NEXT:    v_cvt_f16_f32_e32 v0, v0
; GFX7SELDAG-NEXT:    s_movk_i32 s4, 0x7c00
; GFX7SELDAG-NEXT:    v_and_b32_e32 v0, 0x7fff, v0
; GFX7SELDAG-NEXT:    v_cmp_eq_u32_e32 vcc, s4, v0
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7SELDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isinf_f16:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_mov_b32_e32 v1, 0x204
; GFX8CHECK-NEXT:    v_cmp_class_f16_e32 vcc, v0, v1
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isinf_f16:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_mov_b32_e32 v1, 0x204
; GFX9CHECK-NEXT:    v_cmp_class_f16_e32 vcc, v0, v1
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isinf_f16:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f16_e64 s4, v0, 0x204
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isinf_f16:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f16_e64 s0, v0, 0x204
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f16(half %x, i32 516)  ; 0x204 = "inf"
  ret i1 %1
}

define i1 @isfinite_f16(half %x) nounwind {
; GFX7SELDAG-LABEL: isfinite_f16:
; GFX7SELDAG:       ; %bb.0:
; GFX7SELDAG-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7SELDAG-NEXT:    v_cvt_f16_f32_e32 v0, v0
; GFX7SELDAG-NEXT:    s_movk_i32 s4, 0x7c00
; GFX7SELDAG-NEXT:    v_and_b32_e32 v0, 0x7fff, v0
; GFX7SELDAG-NEXT:    v_cmp_gt_i32_e32 vcc, s4, v0
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7SELDAG-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isfinite_f16:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_mov_b32_e32 v1, 0x1f8
; GFX8CHECK-NEXT:    v_cmp_class_f16_e32 vcc, v0, v1
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isfinite_f16:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_mov_b32_e32 v1, 0x1f8
; GFX9CHECK-NEXT:    v_cmp_class_f16_e32 vcc, v0, v1
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isfinite_f16:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f16_e64 s4, v0, 0x1f8
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isfinite_f16:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f16_e64 s0, v0, 0x1f8
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f16(half %x, i32 504)  ; 0x1f8 = "finite"
  ret i1 %1
}

declare i1 @llvm.is.fpclass.f16(half, i32)
declare <2 x i1> @llvm.is.fpclass.v2f16(<2 x half>, i32)
declare <3 x i1> @llvm.is.fpclass.v3f16(<3 x half>, i32)
declare <4 x i1> @llvm.is.fpclass.v4f16(<4 x half>, i32)
