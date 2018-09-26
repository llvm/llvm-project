; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s


; GCN-LABEL: {{^}}add_var_var_i1:
; GCN: s_xor_b64
define amdgpu_kernel void @add_var_var_i1(i1 addrspace(1)* %out, i1 addrspace(1)* %in0, i1 addrspace(1)* %in1) {
  %a = load volatile i1, i1 addrspace(1)* %in0
  %b = load volatile i1, i1 addrspace(1)* %in1
  %add = add i1 %a, %b
  store i1 %add, i1 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_var_imm_i1:
; GCN: s_not_b64
define amdgpu_kernel void @add_var_imm_i1(i1 addrspace(1)* %out, i1 addrspace(1)* %in) {
  %a = load volatile i1, i1 addrspace(1)* %in
  %add = add i1 %a, 1
  store i1 %add, i1 addrspace(1)* %out
  ret void
}
