; RUN: llc -march=amdgcn -mattr=+promote-alloca,+max-private-element-size-4 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mattr=-promote-alloca,+max-private-element-size-4 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Pointer value is stored in a candidate for LDS usage.

; GCN-LABEL: {{^}}stored_lds_pointer_value:
; GCN: buffer_store_dword v
define amdgpu_kernel void @stored_lds_pointer_value(float addrspace(5)* addrspace(1)* %ptr) #0 {
  %tmp = alloca float, addrspace(5)
  store float 0.0, float  addrspace(5)*%tmp
  store float addrspace(5)* %tmp, float addrspace(5)* addrspace(1)* %ptr
  ret void
}

; GCN-LABEL: {{^}}stored_lds_pointer_value_offset:
; GCN: buffer_store_dword v
define amdgpu_kernel void @stored_lds_pointer_value_offset(float addrspace(5)* addrspace(1)* %ptr) #0 {
  %tmp0 = alloca float, addrspace(5)
  %tmp1 = alloca float, addrspace(5)
  store float 0.0, float  addrspace(5)*%tmp0
  store float 0.0, float  addrspace(5)*%tmp1
  store volatile float addrspace(5)* %tmp0, float addrspace(5)* addrspace(1)* %ptr
  store volatile float addrspace(5)* %tmp1, float addrspace(5)* addrspace(1)* %ptr
  ret void
}

; GCN-LABEL: {{^}}stored_lds_pointer_value_gep:
; GCN-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GCN-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GCN: buffer_store_dword v
; GCN: buffer_store_dword v
define amdgpu_kernel void @stored_lds_pointer_value_gep(float addrspace(5)* addrspace(1)* %ptr, i32 %idx) #0 {
bb:
  %tmp = alloca float, i32 16, addrspace(5)
  store float 0.0, float addrspace(5)* %tmp
  %tmp2 = getelementptr inbounds float, float addrspace(5)* %tmp, i32 %idx
  store float addrspace(5)* %tmp2, float addrspace(5)* addrspace(1)* %ptr
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
define amdgpu_kernel void @stored_vector_pointer_value(i32 addrspace(5)* addrspace(1)* %out, i32 %index) {
entry:
  %tmp0 = alloca [4 x i32], addrspace(5)
  %x = getelementptr inbounds [4 x i32], [4 x i32] addrspace(5)* %tmp0, i32 0, i32 0
  %y = getelementptr inbounds [4 x i32], [4 x i32] addrspace(5)* %tmp0, i32 0, i32 1
  %z = getelementptr inbounds [4 x i32], [4 x i32] addrspace(5)* %tmp0, i32 0, i32 2
  %w = getelementptr inbounds [4 x i32], [4 x i32] addrspace(5)* %tmp0, i32 0, i32 3
  store i32 0, i32 addrspace(5)* %x
  store i32 1, i32 addrspace(5)* %y
  store i32 2, i32 addrspace(5)* %z
  store i32 3, i32 addrspace(5)* %w
  %tmp1 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(5)* %tmp0, i32 0, i32 %index
  store i32 addrspace(5)* %tmp1, i32 addrspace(5)* addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}stored_fi_to_self:
; GCN-NOT: ds_
define amdgpu_kernel void @stored_fi_to_self() #0 {
  %tmp = alloca i32 addrspace(5)*, addrspace(5)
  store volatile i32 addrspace(5)* inttoptr (i32 1234 to i32 addrspace(5)*), i32 addrspace(5)* addrspace(5)* %tmp
  %bitcast = bitcast i32 addrspace(5)* addrspace(5)* %tmp to i32 addrspace(5)*
  store volatile i32 addrspace(5)* %bitcast, i32 addrspace(5)* addrspace(5)* %tmp
  ret void
}

attributes #0 = { nounwind }
