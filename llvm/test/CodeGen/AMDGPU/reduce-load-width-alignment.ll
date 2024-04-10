; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}reduce_i64_load_align_4_width_to_i32:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_and_b32_e32 v{{[0-9]+}}, 0x12d687, [[VAL]]
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @reduce_i64_load_align_4_width_to_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %a = load i64, ptr addrspace(1) %in, align 4
  %and = and i64 %a, 1234567
  store i64 %and, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}reduce_i64_align_4_bitcast_v2i32_elt0:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: buffer_store_dword [[VAL]]
define amdgpu_kernel void @reduce_i64_align_4_bitcast_v2i32_elt0(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %a = load i64, ptr addrspace(1) %in, align 4
  %vec = bitcast i64 %a to <2 x i32>
  %elt0 = extractelement <2 x i32> %vec, i32 0
  store i32 %elt0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}reduce_i64_align_4_bitcast_v2i32_elt1:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:4
; GCN: buffer_store_dword [[VAL]]
define amdgpu_kernel void @reduce_i64_align_4_bitcast_v2i32_elt1(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %a = load i64, ptr addrspace(1) %in, align 4
  %vec = bitcast i64 %a to <2 x i32>
  %elt0 = extractelement <2 x i32> %vec, i32 1
  store i32 %elt0, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
