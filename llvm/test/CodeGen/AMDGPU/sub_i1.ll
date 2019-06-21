; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,WAVE64 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,WAVE32 %s


; GCN-LABEL: {{^}}sub_var_var_i1:
; WAVE32: s_xor_b32
; WAVE64: s_xor_b64
define amdgpu_kernel void @sub_var_var_i1(i1 addrspace(1)* %out, i1 addrspace(1)* %in0, i1 addrspace(1)* %in1) {
  %a = load volatile i1, i1 addrspace(1)* %in0
  %b = load volatile i1, i1 addrspace(1)* %in1
  %sub = sub i1 %a, %b
  store i1 %sub, i1 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sub_var_imm_i1:
; WAVE32: s_not_b32
; WAVE64: s_not_b64
define amdgpu_kernel void @sub_var_imm_i1(i1 addrspace(1)* %out, i1 addrspace(1)* %in) {
  %a = load volatile i1, i1 addrspace(1)* %in
  %sub = sub i1 %a, 1
  store i1 %sub, i1 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sub_i1_cf:
; GCN: ; %endif
; WAVE32: s_not_b32
; WAVE64: s_not_b64
define amdgpu_kernel void @sub_i1_cf(i1 addrspace(1)* %out, i1 addrspace(1)* %a, i1 addrspace(1)* %b) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %d_cmp = icmp ult i32 %tid, 16
  br i1 %d_cmp, label %if, label %else

if:
  %0 = load volatile i1, i1 addrspace(1)* %a
  br label %endif

else:
  %1 = load volatile i1, i1 addrspace(1)* %b
  br label %endif

endif:
  %2 = phi i1 [%0, %if], [%1, %else]
  %3 = sub i1 %2, -1
  store i1 %3, i1 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
