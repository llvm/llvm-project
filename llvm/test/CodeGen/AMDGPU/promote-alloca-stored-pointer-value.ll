; RUN: llc -march=amdgcn -mattr=+promote-alloca,+max-private-element-size-4 -verify-machineinstrs --amdgpu-lower-module-lds-strategy=module < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mattr=-promote-alloca,+max-private-element-size-4 -verify-machineinstrs --amdgpu-lower-module-lds-strategy=module < %s | FileCheck -check-prefix=GCN %s

; Pointer value is stored in a candidate for LDS usage.

; GCN-LABEL: {{^}}stored_lds_pointer_value:
; GCN: buffer_store_dword v
define amdgpu_kernel void @stored_lds_pointer_value(ptr addrspace(1) %ptr) #0 {
  %tmp = alloca float, addrspace(5)
  store float 0.0, ptr  addrspace(5) %tmp
  store ptr addrspace(5) %tmp, ptr addrspace(1) %ptr
  ret void
}

; GCN-LABEL: {{^}}stored_lds_pointer_value_offset:
; GCN: buffer_store_dword v
define amdgpu_kernel void @stored_lds_pointer_value_offset(ptr addrspace(1) %ptr) #0 {
  %tmp0 = alloca float, addrspace(5)
  %tmp1 = alloca float, addrspace(5)
  store float 0.0, ptr  addrspace(5) %tmp0
  store float 0.0, ptr  addrspace(5) %tmp1
  store volatile ptr addrspace(5) %tmp0, ptr addrspace(1) %ptr
  store volatile ptr addrspace(5) %tmp1, ptr addrspace(1) %ptr
  ret void
}

; GCN-LABEL: {{^}}stored_lds_pointer_value_gep:
; GCN-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GCN-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GCN: buffer_store_dword v
; GCN: buffer_store_dword v
define amdgpu_kernel void @stored_lds_pointer_value_gep(ptr addrspace(1) %ptr, i32 %idx) #0 {
bb:
  %tmp = alloca float, i32 16, addrspace(5)
  store float 0.0, ptr addrspace(5) %tmp
  %tmp2 = getelementptr inbounds float, ptr addrspace(5) %tmp, i32 %idx
  store ptr addrspace(5) %tmp2, ptr addrspace(1) %ptr
  ret void
}

; Pointer value is stored in a candidate for vector usage
; GCN-LABEL: {{^}}stored_vector_pointer_value:
; GCN-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GCN-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
define amdgpu_kernel void @stored_vector_pointer_value(ptr addrspace(1) %out, i32 %index) {
entry:
  %tmp0 = alloca [4 x i32], addrspace(5)
  %y = getelementptr inbounds [4 x i32], ptr addrspace(5) %tmp0, i32 0, i32 1
  %z = getelementptr inbounds [4 x i32], ptr addrspace(5) %tmp0, i32 0, i32 2
  %w = getelementptr inbounds [4 x i32], ptr addrspace(5) %tmp0, i32 0, i32 3
  store i32 0, ptr addrspace(5) %tmp0
  store i32 1, ptr addrspace(5) %y
  store i32 2, ptr addrspace(5) %z
  store i32 3, ptr addrspace(5) %w
  %tmp1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %tmp0, i32 0, i32 %index
  store ptr addrspace(5) %tmp1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}stored_fi_to_self:
; GCN-NOT: ds_
define amdgpu_kernel void @stored_fi_to_self() #0 {
  %tmp = alloca ptr addrspace(5), addrspace(5)
  store volatile ptr addrspace(5) inttoptr (i32 1234 to ptr addrspace(5)), ptr addrspace(5) %tmp
  store volatile ptr addrspace(5) %tmp, ptr addrspace(5) %tmp
  ret void
}

attributes #0 = { nounwind }
