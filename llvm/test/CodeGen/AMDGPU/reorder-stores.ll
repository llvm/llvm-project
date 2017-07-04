; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn < %s | FileCheck -check-prefix=SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}no_reorder_v2f64_global_load_store:
; SI: buffer_load_dwordx4
; SI: buffer_load_dwordx4
; SI: buffer_store_dwordx4
; SI: buffer_store_dwordx4
; SI: s_endpgm
define amdgpu_kernel void @no_reorder_v2f64_global_load_store(<2 x double> addrspace(1)* nocapture %x, <2 x double> addrspace(1)* nocapture %y) nounwind {
  %tmp1 = load <2 x double>, <2 x double> addrspace(1)* %x, align 16
  %tmp4 = load <2 x double>, <2 x double> addrspace(1)* %y, align 16
  store <2 x double> %tmp4, <2 x double> addrspace(1)* %x, align 16
  store <2 x double> %tmp1, <2 x double> addrspace(1)* %y, align 16
  ret void
}

; SI-LABEL: {{^}}no_reorder_scalarized_v2f64_local_load_store:
; SI: ds_read2_b64
; SI: ds_write2_b64
; SI: s_endpgm
define amdgpu_kernel void @no_reorder_scalarized_v2f64_local_load_store(<2 x double> addrspace(3)* nocapture %x, <2 x double> addrspace(3)* nocapture %y) nounwind {
  %tmp1 = load <2 x double>, <2 x double> addrspace(3)* %x, align 16
  %tmp4 = load <2 x double>, <2 x double> addrspace(3)* %y, align 16
  store <2 x double> %tmp4, <2 x double> addrspace(3)* %x, align 16
  store <2 x double> %tmp1, <2 x double> addrspace(3)* %y, align 16
  ret void
}

; SI-LABEL: {{^}}no_reorder_split_v8i32_global_load_store:
; SI: buffer_load_dwordx4
; SI: buffer_load_dwordx4
; SI: buffer_load_dwordx4
; SI: buffer_load_dwordx4


; SI: buffer_store_dwordx4
; SI: buffer_store_dwordx4
; SI: buffer_store_dwordx4
; SI: buffer_store_dwordx4
; SI: s_endpgm
define amdgpu_kernel void @no_reorder_split_v8i32_global_load_store(<8 x i32> addrspace(1)* nocapture %x, <8 x i32> addrspace(1)* nocapture %y) nounwind {
  %tmp1 = load <8 x i32>, <8 x i32> addrspace(1)* %x, align 32
  %tmp4 = load <8 x i32>, <8 x i32> addrspace(1)* %y, align 32
  store <8 x i32> %tmp4, <8 x i32> addrspace(1)* %x, align 32
  store <8 x i32> %tmp1, <8 x i32> addrspace(1)* %y, align 32
  ret void
}

; SI-LABEL: {{^}}no_reorder_extload_64:
; SI: ds_read_b64
; SI: ds_read_b64
; SI: ds_write_b64
; SI-NOT: ds_read
; SI: ds_write_b64
; SI: s_endpgm
define amdgpu_kernel void @no_reorder_extload_64(<2 x i32> addrspace(3)* nocapture %x, <2 x i32> addrspace(3)* nocapture %y) nounwind {
  %tmp1 = load <2 x i32>, <2 x i32> addrspace(3)* %x, align 8
  %tmp4 = load <2 x i32>, <2 x i32> addrspace(3)* %y, align 8
  %tmp1ext = zext <2 x i32> %tmp1 to <2 x i64>
  %tmp4ext = zext <2 x i32> %tmp4 to <2 x i64>
  %tmp7 = add <2 x i64> %tmp1ext, <i64 1, i64 1>
  %tmp9 = add <2 x i64> %tmp4ext, <i64 1, i64 1>
  %trunctmp9 = trunc <2 x i64> %tmp9 to <2 x i32>
  %trunctmp7 = trunc <2 x i64> %tmp7 to <2 x i32>
  store <2 x i32> %trunctmp9, <2 x i32> addrspace(3)* %x, align 8
  store <2 x i32> %trunctmp7, <2 x i32> addrspace(3)* %y, align 8
  ret void
}
