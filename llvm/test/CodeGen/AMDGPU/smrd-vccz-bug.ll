; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VCCZ-BUG %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VCCZ-BUG %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=NOVCCZ-BUG %s

; GCN-FUNC: {{^}}vccz_workaround:
; GCN: s_load_dword s{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0x0
; GCN: v_cmp_neq_f32_e64 vcc, s{{[0-9]+}}, 0{{$}}
; GCN: s_waitcnt lgkmcnt(0)
; VCCZ-BUG: s_mov_b64 vcc, vcc
; NOVCCZ-BUG-NOT: s_mov_b64 vcc, vcc
; GCN: s_cbranch_vccnz [[EXIT:[0-9A-Za-z_]+]]
; GCN: buffer_store_dword
; GCN: [[EXIT]]:
; GCN: s_endpgm
define amdgpu_kernel void @vccz_workaround(i32 addrspace(2)* %in, i32 addrspace(1)* %out, float %cond) {
entry:
  %cnd = fcmp oeq float 0.0, %cond
  %sgpr = load volatile i32, i32 addrspace(2)* %in
  br i1 %cnd, label %if, label %endif

if:
  store i32 %sgpr, i32 addrspace(1)* %out
  br label %endif

endif:
  ret void
}

; GCN-FUNC: {{^}}vccz_noworkaround:
; GCN: v_cmp_neq_f32_e32 vcc, 0, v{{[0-9]+}}
; GCN: s_cbranch_vccnz [[EXIT:[0-9A-Za-z_]+]]
; GCN: buffer_store_dword
; GCN: [[EXIT]]:
; GCN: s_endpgm
define amdgpu_kernel void @vccz_noworkaround(float addrspace(1)* %in, float addrspace(1)* %out) {
entry:
  %vgpr = load volatile float, float addrspace(1)* %in
  %cnd = fcmp oeq float 0.0, %vgpr
  br i1 %cnd, label %if, label %endif

if:
  store float %vgpr, float addrspace(1)* %out
  br label %endif

endif:
  ret void
}
