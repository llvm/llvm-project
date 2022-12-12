; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn < %s | FileCheck -check-prefixes=GCN,SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}no_reorder_v2f64_global_load_store:
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN: s_endpgm
define amdgpu_kernel void @no_reorder_v2f64_global_load_store(ptr addrspace(1) nocapture %x, ptr addrspace(1) nocapture %y) nounwind {
  %tmp1 = load <2 x double>, ptr addrspace(1) %x, align 16
  %tmp4 = load <2 x double>, ptr addrspace(1) %y, align 16
  store <2 x double> %tmp4, ptr addrspace(1) %x, align 16
  store <2 x double> %tmp1, ptr addrspace(1) %y, align 16
  ret void
}

; GCN-LABEL: {{^}}no_reorder_scalarized_v2f64_local_load_store:
; SI: ds_read2_b64
; SI: ds_write2_b64

; VI: ds_read_b128
; VI: ds_write_b128

; GCN: s_endpgm
define amdgpu_kernel void @no_reorder_scalarized_v2f64_local_load_store(ptr addrspace(3) nocapture %x, ptr addrspace(3) nocapture %y) nounwind {
  %tmp1 = load <2 x double>, ptr addrspace(3) %x, align 16
  %tmp4 = load <2 x double>, ptr addrspace(3) %y, align 16
  store <2 x double> %tmp4, ptr addrspace(3) %x, align 16
  store <2 x double> %tmp1, ptr addrspace(3) %y, align 16
  ret void
}

; GCN-LABEL: {{^}}no_reorder_split_v8i32_global_load_store:
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4


; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN: s_endpgm
define amdgpu_kernel void @no_reorder_split_v8i32_global_load_store(ptr addrspace(1) nocapture %x, ptr addrspace(1) nocapture %y) nounwind {
  %tmp1 = load <8 x i32>, ptr addrspace(1) %x, align 32
  %tmp4 = load <8 x i32>, ptr addrspace(1) %y, align 32
  store <8 x i32> %tmp4, ptr addrspace(1) %x, align 32
  store <8 x i32> %tmp1, ptr addrspace(1) %y, align 32
  ret void
}

; GCN-LABEL: {{^}}no_reorder_extload_64:
; GCN: ds_read_b64
; GCN: ds_read_b64
; GCN: ds_write_b64
; GCN-NOT: ds_read
; GCN: ds_write_b64
; GCN: s_endpgm
define amdgpu_kernel void @no_reorder_extload_64(ptr addrspace(3) nocapture %x, ptr addrspace(3) nocapture %y) nounwind {
  %tmp1 = load <2 x i32>, ptr addrspace(3) %x, align 8
  %tmp4 = load <2 x i32>, ptr addrspace(3) %y, align 8
  %tmp1ext = zext <2 x i32> %tmp1 to <2 x i64>
  %tmp4ext = zext <2 x i32> %tmp4 to <2 x i64>
  %tmp7 = add <2 x i64> %tmp1ext, <i64 1, i64 1>
  %tmp9 = add <2 x i64> %tmp4ext, <i64 1, i64 1>
  %trunctmp9 = trunc <2 x i64> %tmp9 to <2 x i32>
  %trunctmp7 = trunc <2 x i64> %tmp7 to <2 x i32>
  store <2 x i32> %trunctmp9, ptr addrspace(3) %x, align 8
  store <2 x i32> %trunctmp7, ptr addrspace(3) %y, align 8
  ret void
}
