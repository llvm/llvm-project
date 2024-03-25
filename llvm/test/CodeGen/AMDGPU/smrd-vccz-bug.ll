; RUN: llc -mtriple=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VCCZ-BUG %s
; RUN: llc -mtriple=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VCCZ-BUG %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}vccz_workaround:
; GCN: s_load_dword [[REG:s[0-9]+]], s[{{[0-9]+:[0-9]+}}],
; GCN: v_cmp_neq_f32_e64 {{[^,]*}}, [[REG]], 0{{$}}
; VCCZ-BUG: s_waitcnt lgkmcnt(0)
; VCCZ-BUG: s_mov_b64 vcc, vcc
; GCN-NOT: s_mov_b64 vcc, vcc
; GCN: s_cbranch_vccnz [[EXIT:.L[0-9A-Za-z_]+]]
; GCN: buffer_store_dword
; GCN: [[EXIT]]:
; GCN: s_endpgm
define amdgpu_kernel void @vccz_workaround(ptr addrspace(4) %in, ptr addrspace(1) %out, float %cond) {
entry:
  %cnd = fcmp oeq float 0.0, %cond
  %sgpr = load volatile i32, ptr addrspace(4) %in
  br i1 %cnd, label %if, label %endif

if:
  store i32 %sgpr, ptr addrspace(1) %out
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}vccz_noworkaround:
; GCN: v_cmp_neq_f32_e32 vcc, 0, v{{[0-9]+}}
; GCN-NOT: s_waitcnt lgkmcnt(0)
; GCN-NOT: s_mov_b64 vcc, vcc
; GCN: s_cbranch_vccnz [[EXIT:.L[0-9A-Za-z_]+]]
; GCN: buffer_store_dword
; GCN: [[EXIT]]:
; GCN: s_endpgm
define amdgpu_kernel void @vccz_noworkaround(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %vgpr = load volatile float, ptr addrspace(1) %in
  %cnd = fcmp oeq float 0.0, %vgpr
  br i1 %cnd, label %if, label %endif

if:
  store float %vgpr, ptr addrspace(1) %out
  br label %endif

endif:
  ret void
}
