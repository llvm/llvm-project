; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10,FUNC %s

; FUNC-LABEL: {{^}}s_add_i32:
; GCN: s_add_i32 s[[REG:[0-9]+]], {{s[0-9]+, s[0-9]+}}
; GCN: v_mov_b32_e32 v[[V_REG:[0-9]+]], s[[REG]]
; GCN: buffer_store_{{dword|b32}} v[[V_REG]],
define amdgpu_kernel void @s_add_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %b_ptr = getelementptr i32, ptr addrspace(1) %in, i32 1
  %a = load i32, ptr addrspace(1) %in
  %b = load i32, ptr addrspace(1) %b_ptr
  %result = add i32 %a, %b
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}s_add_v2i32:
; GCN: s_add_i32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
; GCN: s_add_i32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
define amdgpu_kernel void @s_add_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %b_ptr = getelementptr <2 x i32>, ptr addrspace(1) %in, i32 1
  %a = load <2 x i32>, ptr addrspace(1) %in
  %b = load <2 x i32>, ptr addrspace(1) %b_ptr
  %result = add <2 x i32> %a, %b
  store <2 x i32> %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}s_add_v4i32:
; GCN: s_add_i32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
; GCN: s_add_i32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
; GCN: s_add_i32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
; GCN: s_add_i32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
define amdgpu_kernel void @s_add_v4i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %b_ptr = getelementptr <4 x i32>, ptr addrspace(1) %in, i32 1
  %a = load <4 x i32>, ptr addrspace(1) %in
  %b = load <4 x i32>, ptr addrspace(1) %b_ptr
  %result = add <4 x i32> %a, %b
  store <4 x i32> %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}s_add_v8i32:
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
define amdgpu_kernel void @s_add_v8i32(ptr addrspace(1) %out, <8 x i32> %a, <8 x i32> %b) {
entry:
  %0 = add <8 x i32> %a, %b
  store <8 x i32> %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}s_add_v16i32:
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
define amdgpu_kernel void @s_add_v16i32(ptr addrspace(1) %out, <16 x i32> %a, <16 x i32> %b) {
entry:
  %0 = add <16 x i32> %a, %b
  store <16 x i32> %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v_add_i32:
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[A:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[B:v[0-9]+]]
; SIVI: v_add_{{i|u}}32_e32 v{{[0-9]+}}, vcc, [[A]], [[B]]
; GFX9: v_add_u32_e32 v{{[0-9]+}}, [[A]], [[B]]
; GFX10: v_add_nc_u32_e32 v{{[0-9]+}}, [[A]], [[B]]
define amdgpu_kernel void @v_add_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %in, i32 %tid
  %b_ptr = getelementptr i32, ptr addrspace(1) %gep, i32 1
  %a = load volatile i32, ptr addrspace(1) %gep
  %b = load volatile i32, ptr addrspace(1) %b_ptr
  %result = add i32 %a, %b
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v_add_imm_i32:
; GCN: {{buffer|flat|global}}_load_{{dword|b32}} [[A:v[0-9]+]]
; SIVI: v_add_{{i|u}}32_e32 v{{[0-9]+}}, vcc, 0x7b, [[A]]
; GFX9: v_add_u32_e32 v{{[0-9]+}}, 0x7b, [[A]]
; GFX10: v_add_nc_u32_e32 v{{[0-9]+}}, 0x7b, [[A]]
define amdgpu_kernel void @v_add_imm_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(1) %in, i32 %tid
  %b_ptr = getelementptr i32, ptr addrspace(1) %gep, i32 1
  %a = load volatile i32, ptr addrspace(1) %gep
  %result = add i32 %a, 123
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}add64:
; GCN: s_add_u32
; GCN: s_addc_u32
define amdgpu_kernel void @add64(ptr addrspace(1) %out, i64 %a, i64 %b) {
entry:
  %add = add i64 %a, %b
  store i64 %add, ptr addrspace(1) %out
  ret void
}

; The v_addc_u32 and v_add_i32 instruction can't read SGPRs, because they
; use VCC.  The test is designed so that %a will be stored in an SGPR and
; %0 will be stored in a VGPR, so the comiler will be forced to copy %a
; to a VGPR before doing the add.

; FUNC-LABEL: {{^}}add64_sgpr_vgpr:
; GCN-NOT: v_addc_u32_e32 s
define amdgpu_kernel void @add64_sgpr_vgpr(ptr addrspace(1) %out, i64 %a, ptr addrspace(1) %in) {
entry:
  %0 = load i64, ptr addrspace(1) %in
  %1 = add i64 %a, %0
  store i64 %1, ptr addrspace(1) %out
  ret void
}

; Test i64 add inside a branch.
; FUNC-LABEL: {{^}}add64_in_branch:
; GCN: s_add_u32
; GCN: s_addc_u32
define amdgpu_kernel void @add64_in_branch(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %a, i64 %b, i64 %c) {
entry:
  %0 = icmp eq i64 %a, 0
  br i1 %0, label %if, label %else

if:
  %1 = load i64, ptr addrspace(1) %in
  br label %endif

else:
  %2 = add i64 %a, %b
  br label %endif

endif:
  %3 = phi i64 [%1, %if], [%2, %else]
  store i64 %3, ptr addrspace(1) %out
  ret void
}

; Make sure the VOP3 form of add is initially selected. Otherwise pair
; of opies from/to VCC would be necessary

; GCN-LABEL: {{^}}add_select_vop3:
; SI: v_add_i32_e64 v0, s[0:1], s0, v0
; VI: v_add_u32_e64 v0, s[0:1], s0, v0
; GFX9: v_add_u32_e32 v0, s0, v0
; GFX10: v_add_nc_u32_e32 v0, s0, v0

; GCN: ; def vcc
; GCN: ds_{{write|store}}_b32
; GCN: ; use vcc
define amdgpu_ps void @add_select_vop3(i32 inreg %s, i32 %v) {
  %vcc = call i64 asm sideeffect "; def vcc", "={vcc}"()
  %sub = add i32 %v, %s
  store i32 %sub, ptr addrspace(3) undef
  call void asm sideeffect "; use vcc", "{vcc}"(i64 %vcc)
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
