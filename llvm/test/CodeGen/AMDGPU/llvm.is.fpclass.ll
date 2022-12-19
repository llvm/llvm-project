; RUN:  llc -global-isel=0 -march=amdgcn -mcpu=gfx704 -verify-machineinstrs < %s  | FileCheck --check-prefixes=GFX7SELDAG,GFX7CHECK %s
; RUN:  llc -global-isel=1 -march=amdgcn -mcpu=gfx704 -verify-machineinstrs < %s  | FileCheck --check-prefixes=GFX7GLISEL,GFX7CHECK %s
; RUN:  llc -global-isel=0 -march=amdgcn -mcpu=gfx803 -verify-machineinstrs < %s  | FileCheck --check-prefixes=GFX8SELDAG,GFX8CHECK %s
; RUN:  llc -global-isel=1 -march=amdgcn -mcpu=gfx803 -verify-machineinstrs < %s  | FileCheck --check-prefixes=GFX8GLISEL,GFX8CHECK %s
; RUN:  llc -global-isel=0 -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s  | FileCheck --check-prefixes=GFX9CHECK %s
; RUN:  llc -global-isel=1 -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s  | FileCheck --check-prefixes=GFX9CHECK %s
; RUN:  llc -global-isel=0 -march=amdgcn -mcpu=gfx1031 -verify-machineinstrs < %s | FileCheck --check-prefixes=GFX10CHECK %s
; RUN:  llc -global-isel=1 -march=amdgcn -mcpu=gfx1031 -verify-machineinstrs < %s | FileCheck --check-prefixes=GFX10CHECK %s
; RUN:  llc -global-isel=0 -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck --check-prefix=GFX11CHECK %s
; RUN:  llc -global-isel=1 -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck --check-prefix=GFX11CHECK %s

define amdgpu_kernel void @sgpr_isnan_f32(ptr addrspace(1) %out, float %x) {
; GFX7SELDAG-LABEL: sgpr_isnan_f32:
; GFX7SELDAG:       ; %bb.0:
; GFX7SELDAG-NEXT:    s_load_dword s4, s[0:1], 0xb
; GFX7SELDAG-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; GFX7SELDAG-NEXT:    s_mov_b32 s3, 0xf000
; GFX7SELDAG-NEXT:    s_mov_b32 s2, -1
; GFX7SELDAG-NEXT:    s_waitcnt lgkmcnt(0)
; GFX7SELDAG-NEXT:    v_cmp_class_f32_e64 s[4:5], s4, 3
; GFX7SELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, -1, s[4:5]
; GFX7SELDAG-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GFX7SELDAG-NEXT:    s_endpgm
;
; GFX7GLISEL-LABEL: sgpr_isnan_f32:
; GFX7GLISEL:       ; %bb.0:
; GFX7GLISEL-NEXT:    s_load_dword s3, s[0:1], 0xb
; GFX7GLISEL-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x9
; GFX7GLISEL-NEXT:    s_mov_b32 s2, -1
; GFX7GLISEL-NEXT:    s_waitcnt lgkmcnt(0)
; GFX7GLISEL-NEXT:    v_cmp_class_f32_e64 s[4:5], s3, 3
; GFX7GLISEL-NEXT:    v_cndmask_b32_e64 v0, 0, -1, s[4:5]
; GFX7GLISEL-NEXT:    s_mov_b32 s3, 0xf000
; GFX7GLISEL-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GFX7GLISEL-NEXT:    s_endpgm
;
; GFX8CHECK-LABEL: sgpr_isnan_f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_load_dword s2, s[0:1], 0x2c
; GFX8CHECK-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; GFX8CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[2:3], s2, 3
; GFX8CHECK-NEXT:    v_mov_b32_e32 v0, s0
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, -1, s[2:3]
; GFX8CHECK-NEXT:    v_mov_b32_e32 v1, s1
; GFX8CHECK-NEXT:    flat_store_dword v[0:1], v2
; GFX8CHECK-NEXT:    s_endpgm
;
; GFX9CHECK-LABEL: sgpr_isnan_f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_load_dword s4, s[0:1], 0x2c
; GFX9CHECK-NEXT:    s_load_dwordx2 s[2:3], s[0:1], 0x24
; GFX9CHECK-NEXT:    v_mov_b32_e32 v0, 0
; GFX9CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[0:1], s4, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, -1, s[0:1]
; GFX9CHECK-NEXT:    global_store_dword v0, v1, s[2:3]
; GFX9CHECK-NEXT:    s_endpgm
;
; GFX10CHECK-LABEL: sgpr_isnan_f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_clause 0x1
; GFX10CHECK-NEXT:    s_load_dword s2, s[0:1], 0x2c
; GFX10CHECK-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; GFX10CHECK-NEXT:    v_mov_b32_e32 v0, 0
; GFX10CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s2, s2, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, -1, s2
; GFX10CHECK-NEXT:    global_store_dword v0, v1, s[0:1]
; GFX10CHECK-NEXT:    s_endpgm
;
; GFX11CHECK-LABEL: sgpr_isnan_f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_clause 0x1
; GFX11CHECK-NEXT:    s_load_b32 s2, s[0:1], 0x2c
; GFX11CHECK-NEXT:    s_load_b64 s[0:1], s[0:1], 0x24
; GFX11CHECK-NEXT:    v_mov_b32_e32 v0, 0
; GFX11CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s2, s2, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, -1, s2
; GFX11CHECK-NEXT:    global_store_b32 v0, v1, s[0:1]
; GFX11CHECK-NEXT:    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
; GFX11CHECK-NEXT:    s_endpgm
  %result = call i1 @llvm.is.fpclass.f32(float %x, i32 3)  ; nan
  %sext = sext i1 %result to i32
  store i32 %sext, ptr addrspace(1) %out, align 4
  ret void
}

define amdgpu_kernel void @sgpr_isnan_f64(ptr addrspace(1) %out, double %x) {
; GFX7ISELDAG-LABEL: sgpr_isnan_f64:
; GFX7ISELDAG:       ; %bb.0:
; GFX7ISELDAG-NEXT:    s_load_dwordx4 s[0:3], s[0:1], 0x9
; GFX7ISELDAG-NEXT:    s_mov_b32 s7, 0xf000
; GFX7ISELDAG-NEXT:    s_mov_b32 s6, -1
; GFX7ISELDAG-NEXT:    s_waitcnt lgkmcnt(0)
; GFX7ISELDAG-NEXT:    s_mov_b32 s4, s0
; GFX7ISELDAG-NEXT:    s_mov_b32 s5, s1
; GFX7ISELDAG-NEXT:    v_cmp_class_f64_e64 s[0:1], s[2:3], 3
; GFX7ISELDAG-NEXT:    v_cndmask_b32_e64 v0, 0, -1, s[0:1]
; GFX7ISELDAG-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; GFX7ISELDAG-NEXT:    s_endpgm
;
; GFX7GLISEL-LABEL: sgpr_isnan_f64:
; GFX7GLISEL:       ; %bb.0:
; GFX7GLISEL-NEXT:    s_load_dwordx4 s[0:3], s[0:1], 0x9
; GFX7GLISEL-NEXT:    s_waitcnt lgkmcnt(0)
; GFX7GLISEL-NEXT:    v_cmp_class_f64_e64 s[2:3], s[2:3], 3
; GFX7GLISEL-NEXT:    v_cndmask_b32_e64 v0, 0, -1, s[2:3]
; GFX7GLISEL-NEXT:    s_mov_b32 s2, -1
; GFX7GLISEL-NEXT:    s_mov_b32 s3, 0xf000
; GFX7GLISEL-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; GFX7GLISEL-NEXT:    s_endpgm
;
; GFX8SELDAG-LABEL: sgpr_isnan_f64:
; GFX8SELDAG:       ; %bb.0:
; GFX8SELDAG-NEXT:    s_load_dwordx4 s[0:3], s[0:1], 0x24
; GFX8SELDAG-NEXT:    s_waitcnt lgkmcnt(0)
; GFX8SELDAG-NEXT:    v_mov_b32_e32 v0, s0
; GFX8SELDAG-NEXT:    v_mov_b32_e32 v1, s1
; GFX8SELDAG-NEXT:    v_cmp_class_f64_e64 s[0:1], s[2:3], 3
; GFX8SELDAG-NEXT:    v_cndmask_b32_e64 v2, 0, -1, s[0:1]
; GFX8SELDAG-NEXT:    flat_store_dword v[0:1], v2
; GFX8SELDAG-NEXT:    s_endpgm
;
; GFX8GLISEL-LABEL: sgpr_isnan_f64:
; GFX8GLISEL:       ; %bb.0:
; GFX8GLISEL-NEXT:    s_load_dwordx4 s[0:3], s[0:1], 0x24
; GFX8GLISEL-NEXT:    s_waitcnt lgkmcnt(0)
; GFX8GLISEL-NEXT:    v_cmp_class_f64_e64 s[2:3], s[2:3], 3
; GFX8GLISEL-NEXT:    v_mov_b32_e32 v0, s0
; GFX8GLISEL-NEXT:    v_mov_b32_e32 v1, s1
; GFX8GLISEL-NEXT:    v_cndmask_b32_e64 v2, 0, -1, s[2:3]
; GFX8GLISEL-NEXT:    flat_store_dword v[0:1], v2
; GFX8GLISEL-NEXT:    s_endpgm
;
; GFX9CHECK-LABEL: sgpr_isnan_f64:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_load_dwordx4 s[0:3], s[0:1], 0x24
; GFX9CHECK-NEXT:    v_mov_b32_e32 v0, 0
; GFX9CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f64_e64 s[2:3], s[2:3], 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, -1, s[2:3]
; GFX9CHECK-NEXT:    global_store_dword v0, v1, s[0:1]
; GFX9CHECK-NEXT:    s_endpgm
;
; GFX10CHECK-LABEL: sgpr_isnan_f64:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_load_dwordx4 s[0:3], s[0:1], 0x24
; GFX10CHECK-NEXT:    v_mov_b32_e32 v0, 0
; GFX10CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; GFX10CHECK-NEXT:    v_cmp_class_f64_e64 s2, s[2:3], 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, -1, s2
; GFX10CHECK-NEXT:    global_store_dword v0, v1, s[0:1]
; GFX10CHECK-NEXT:    s_endpgm
;
; GFX11CHECK-LABEL: sgpr_isnan_f64:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_load_b128 s[0:3], s[0:1], 0x24
; GFX11CHECK-NEXT:    v_mov_b32_e32 v0, 0
; GFX11CHECK-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11CHECK-NEXT:    v_cmp_class_f64_e64 s2, s[2:3], 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, -1, s2
; GFX11CHECK-NEXT:    global_store_b32 v0, v1, s[0:1]
; GFX11CHECK-NEXT:    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
; GFX11CHECK-NEXT:    s_endpgm
  %result = call i1 @llvm.is.fpclass.f64(double %x, i32 3)  ; nan
  %sext = sext i1 %result to i32
  store i32 %sext, ptr addrspace(1) %out, align 4
  ret void
}

define i1 @isnan_f32(float %x) nounwind {
; GFX7CHECK-LABEL: isnan_f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f32(float %x, i32 3)  ; nan
  ret i1 %1
}

define <2 x i1> @isnan_v2f32(<2 x float> %x) nounwind {
; GFX7CHECK-LABEL: isnan_v2f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_v2f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_v2f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_v2f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v1, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_v2f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v1, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %x, i32 3)  ; nan
  ret <2 x i1> %1
}

define <3 x i1> @isnan_v3f32(<3 x float> %x) nounwind {
; GFX7CHECK-LABEL: isnan_v3f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_v3f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_v3f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_v3f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v1, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v2, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_v3f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v1, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v2, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call <3 x i1> @llvm.is.fpclass.v3f32(<3 x float> %x, i32 3)  ; nan
  ret <3 x i1> %1
}

define <4 x i1> @isnan_v4f32(<4 x float> %x) nounwind {
; GFX7CHECK-LABEL: isnan_v4f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_v4f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_v4f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_v4f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v1, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v2, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v3, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_v4f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v1, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v2, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v3, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call <4 x i1> @llvm.is.fpclass.v4f32(<4 x float> %x, i32 3)  ; nan
  ret <4 x i1> %1
}

define <5 x i1> @isnan_v5f32(<5 x float> %x) nounwind {
; GFX7CHECK-LABEL: isnan_v5f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_v5f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_v5f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_v5f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v1, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v2, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v3, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v4, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_v5f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v1, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v2, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v3, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v4, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call <5 x i1> @llvm.is.fpclass.v5f32(<5 x float> %x, i32 3)  ; nan
  ret <5 x i1> %1
}

define <6 x i1> @isnan_v6f32(<6 x float> %x) nounwind {
; GFX7CHECK-LABEL: isnan_v6f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v5, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_v6f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v5, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_v6f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v5, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_v6f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v1, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v2, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v3, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v4, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v5, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_v6f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v1, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v2, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v3, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v4, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v5, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call <6 x i1> @llvm.is.fpclass.v6f32(<6 x float> %x, i32 3)  ; nan
  ret <6 x i1> %1
}

define <7 x i1> @isnan_v7f32(<7 x float> %x) nounwind {
; GFX7CHECK-LABEL: isnan_v7f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v5, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v6, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_v7f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v5, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v6, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_v7f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v5, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v6, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_v7f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v1, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v2, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v3, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v4, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v5, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v6, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_v7f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v1, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v2, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v3, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v4, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v5, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v6, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call <7 x i1> @llvm.is.fpclass.v7f32(<7 x float> %x, i32 3)  ; nan
  ret <7 x i1> %1
}

define <8 x i1> @isnan_v8f32(<8 x float> %x) nounwind {
; GFX7CHECK-LABEL: isnan_v8f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v5, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v6, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v7, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v7, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_v8f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v5, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v6, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v7, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v7, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_v8f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v5, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v6, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v7, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v7, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_v8f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v1, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v2, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v3, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v4, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v5, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v6, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v7, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v7, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_v8f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v1, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v2, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v3, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v4, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v5, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v6, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v7, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v7, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call <8 x i1> @llvm.is.fpclass.v8f32(<8 x float> %x, i32 3)  ; nan
  ret <8 x i1> %1
}

define <16 x i1> @isnan_v16f32(<16 x float> %x) nounwind {
; GFX7CHECK-LABEL: isnan_v16f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v5, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v6, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v7, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v7, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v8, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v8, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v9, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v9, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v10, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v10, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v11, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v11, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v12, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v12, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v13, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v13, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v14, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v14, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v15, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v15, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_v16f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v5, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v6, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v7, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v7, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v8, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v8, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v9, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v9, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v10, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v10, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v11, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v11, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v12, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v12, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v13, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v13, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v14, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v14, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v15, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v15, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_v16f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v1, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v2, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v3, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v4, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v5, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v6, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v7, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v7, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v8, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v8, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v9, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v9, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v10, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v10, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v11, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v11, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v12, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v12, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v13, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v13, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v14, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v14, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v15, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v15, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_v16f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v1, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v2, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v3, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v4, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v5, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v6, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v7, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v7, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v8, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v8, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v9, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v9, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v10, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v10, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v11, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v11, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v12, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v12, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v13, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v13, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v14, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v14, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v15, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v15, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_v16f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v1, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v2, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v3, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v3, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v4, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v4, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v5, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v5, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v6, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v6, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v7, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v7, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v8, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v8, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v9, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v9, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v10, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v10, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v11, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v11, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v12, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v12, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v13, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v13, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v14, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v14, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v15, 3
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v15, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call <16 x i1> @llvm.is.fpclass.v16f32(<16 x float> %x, i32 3)  ; nan
  ret <16 x i1> %1
}

define i1 @isnan_f64(double %x) nounwind {
; GFX7CHECK-LABEL: isnan_f64:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_cmp_class_f64_e64 s[4:5], v[0:1], 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_f64:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f64_e64 s[4:5], v[0:1], 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_f64:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f64_e64 s[4:5], v[0:1], 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_f64:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f64_e64 s4, v[0:1], 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_f64:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f64_e64 s0, v[0:1], 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f64(double %x, i32 3)  ; nan
  ret i1 %1
}

define i1 @isnan_f32_strictfp(float %x) strictfp nounwind {
; GFX7CHECK-LABEL: isnan_f32_strictfp:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_f32_strictfp:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_f32_strictfp:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f32_e64 s[4:5], v0, 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_f32_strictfp:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_f32_strictfp:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f32(float %x, i32 3)  ; nan
  ret i1 %1
}

define i1 @isnan_f64_strictfp(double %x) strictfp nounwind {
; GFX7CHECK-LABEL: isnan_f64_strictfp:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_cmp_class_f64_e64 s[4:5], v[0:1], 3
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnan_f64_strictfp:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_cmp_class_f64_e64 s[4:5], v[0:1], 3
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnan_f64_strictfp:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_cmp_class_f64_e64 s[4:5], v[0:1], 3
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnan_f64_strictfp:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f64_e64 s4, v[0:1], 3
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnan_f64_strictfp:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f64_e64 s0, v[0:1], 3
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f64(double %x, i32 3)  ; nan
  ret i1 %1
}

define i1 @isinf_f32(float %x) nounwind {
; GFX7CHECK-LABEL: isinf_f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_mov_b32_e32 v1, 0x204
; GFX7CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isinf_f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_mov_b32_e32 v1, 0x204
; GFX8CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isinf_f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_mov_b32_e32 v1, 0x204
; GFX9CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isinf_f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 0x204
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isinf_f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 0x204
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f32(float %x, i32 516)  ; 0x204 = "inf"
  ret i1 %1
}

define i1 @isinf_f64(double %x) nounwind {
; GFX7CHECK-LABEL: isinf_f64:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_mov_b32_e32 v2, 0x204
; GFX7CHECK-NEXT:    v_cmp_class_f64_e32 vcc, v[0:1], v2
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isinf_f64:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_mov_b32_e32 v2, 0x204
; GFX8CHECK-NEXT:    v_cmp_class_f64_e32 vcc, v[0:1], v2
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isinf_f64:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_mov_b32_e32 v2, 0x204
; GFX9CHECK-NEXT:    v_cmp_class_f64_e32 vcc, v[0:1], v2
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isinf_f64:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f64_e64 s4, v[0:1], 0x204
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isinf_f64:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f64_e64 s0, v[0:1], 0x204
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f64(double %x, i32 516)  ; 0x204 = "inf"
  ret i1 %1
}

define i1 @isfinite_f32(float %x) nounwind {
; GFX7CHECK-LABEL: isfinite_f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_mov_b32_e32 v1, 0x1f8
; GFX7CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isfinite_f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_mov_b32_e32 v1, 0x1f8
; GFX8CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isfinite_f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_mov_b32_e32 v1, 0x1f8
; GFX9CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isfinite_f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 0x1f8
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isfinite_f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 0x1f8
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f32(float %x, i32 504)  ; 0x1f8 = "finite"
  ret i1 %1
}

define i1 @isfinite_f64(double %x) nounwind {
; GFX7CHECK-LABEL: isfinite_f64:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_mov_b32_e32 v2, 0x1f8
; GFX7CHECK-NEXT:    v_cmp_class_f64_e32 vcc, v[0:1], v2
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isfinite_f64:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_mov_b32_e32 v2, 0x1f8
; GFX8CHECK-NEXT:    v_cmp_class_f64_e32 vcc, v[0:1], v2
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isfinite_f64:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_mov_b32_e32 v2, 0x1f8
; GFX9CHECK-NEXT:    v_cmp_class_f64_e32 vcc, v[0:1], v2
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isfinite_f64:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f64_e64 s4, v[0:1], 0x1f8
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isfinite_f64:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f64_e64 s0, v[0:1], 0x1f8
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f64(double %x, i32 504)  ; 0x1f8 = "finite"
  ret i1 %1
}

define i1 @isnormal_f32(float %x) nounwind {
; GFX7CHECK-LABEL: isnormal_f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_mov_b32_e32 v1, 0x108
; GFX7CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnormal_f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_mov_b32_e32 v1, 0x108
; GFX8CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnormal_f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_mov_b32_e32 v1, 0x108
; GFX9CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnormal_f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 0x108
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnormal_f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 0x108
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f32(float %x, i32 264)  ; 0x108 = "normal"
  ret i1 %1
}

define <2 x i1> @isnormal_v2f64(<2 x double> %x) nounwind {
; GFX7CHECK-LABEL: isnormal_v2f64:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_mov_b32_e32 v4, 0x108
; GFX7CHECK-NEXT:    v_cmp_class_f64_e32 vcc, v[0:1], v4
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7CHECK-NEXT:    v_cmp_class_f64_e32 vcc, v[2:3], v4
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, vcc
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: isnormal_v2f64:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_mov_b32_e32 v4, 0x108
; GFX8CHECK-NEXT:    v_cmp_class_f64_e32 vcc, v[0:1], v4
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX8CHECK-NEXT:    v_cmp_class_f64_e32 vcc, v[2:3], v4
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, vcc
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: isnormal_v2f64:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_mov_b32_e32 v4, 0x108
; GFX9CHECK-NEXT:    v_cmp_class_f64_e32 vcc, v[0:1], v4
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX9CHECK-NEXT:    v_cmp_class_f64_e32 vcc, v[2:3], v4
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, vcc
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: isnormal_v2f64:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f64_e64 s4, v[0:1], 0x108
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    v_cmp_class_f64_e64 s4, v[2:3], 0x108
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: isnormal_v2f64:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f64_e64 s0, v[0:1], 0x108
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    v_cmp_class_f64_e64 s0, v[2:3], 0x108
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call <2 x i1> @llvm.is.fpclass.v2f64(<2 x double> %x, i32 264)  ; 0x108 = "normal"
  ret <2 x i1> %1
}

define i1 @issubnormal_f32(float %x) nounwind {
; GFX7CHECK-LABEL: issubnormal_f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_mov_b32_e32 v1, 0x90
; GFX7CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: issubnormal_f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_mov_b32_e32 v1, 0x90
; GFX8CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: issubnormal_f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_mov_b32_e32 v1, 0x90
; GFX9CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: issubnormal_f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 0x90
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: issubnormal_f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 0x90
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f32(float %x, i32 144)  ; 0x90 = "subnormal"
  ret i1 %1
}

define i1 @iszero_f32(float %x) nounwind {
; GFX7CHECK-LABEL: iszero_f32:
; GFX7CHECK:       ; %bb.0:
; GFX7CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7CHECK-NEXT:    v_mov_b32_e32 v1, 0x60
; GFX7CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX7CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX7CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8CHECK-LABEL: iszero_f32:
; GFX8CHECK:       ; %bb.0:
; GFX8CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8CHECK-NEXT:    v_mov_b32_e32 v1, 0x60
; GFX8CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX8CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX8CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9CHECK-LABEL: iszero_f32:
; GFX9CHECK:       ; %bb.0:
; GFX9CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9CHECK-NEXT:    v_mov_b32_e32 v1, 0x60
; GFX9CHECK-NEXT:    v_cmp_class_f32_e32 vcc, v0, v1
; GFX9CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, vcc
; GFX9CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10CHECK-LABEL: iszero_f32:
; GFX10CHECK:       ; %bb.0:
; GFX10CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10CHECK-NEXT:    v_cmp_class_f32_e64 s4, v0, 0x60
; GFX10CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s4
; GFX10CHECK-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11CHECK-LABEL: iszero_f32:
; GFX11CHECK:       ; %bb.0:
; GFX11CHECK-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11CHECK-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX11CHECK-NEXT:    v_cmp_class_f32_e64 s0, v0, 0x60
; GFX11CHECK-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX11CHECK-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s0
; GFX11CHECK-NEXT:    s_setpc_b64 s[30:31]
  %1 = call i1 @llvm.is.fpclass.f32(float %x, i32 96)  ; 0x60 = "zero"
  ret i1 %1
}

declare i1 @llvm.is.fpclass.f32(float, i32)
declare i1 @llvm.is.fpclass.f64(double, i32)
declare <2 x i1> @llvm.is.fpclass.v2f32(<2 x float>, i32)
declare <3 x i1> @llvm.is.fpclass.v3f32(<3 x float>, i32)
declare <4 x i1> @llvm.is.fpclass.v4f32(<4 x float>, i32)
declare <5 x i1> @llvm.is.fpclass.v5f32(<5 x float>, i32)
declare <6 x i1> @llvm.is.fpclass.v6f32(<6 x float>, i32)
declare <7 x i1> @llvm.is.fpclass.v7f32(<7 x float>, i32)
declare <8 x i1> @llvm.is.fpclass.v8f32(<8 x float>, i32)
declare <16 x i1> @llvm.is.fpclass.v16f32(<16 x float>, i32)
declare <2 x i1> @llvm.is.fpclass.v2f64(<2 x double>, i32)
