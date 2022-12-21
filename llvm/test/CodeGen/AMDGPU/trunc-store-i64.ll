; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}trunc_store_v4i64_v4i8:
; GCN: global_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @trunc_store_v4i64_v4i8(ptr addrspace(1) %out, <4 x i64> %in) {
entry:
  %trunc = trunc <4 x i64> %in to < 4 x i8>
  store <4 x i8> %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_store_v8i64_v8i8:
; GCN: global_store_dwordx2 v{{[0-9]+}}, v{{\[[0-9]:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @trunc_store_v8i64_v8i8(ptr addrspace(1) %out, <8 x i64> %in) {
entry:
  %trunc = trunc <8 x i64> %in to < 8 x i8>
  store <8 x i8> %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_store_v8i64_v8i16:
; GCN: global_store_dwordx4 v{{[0-9]+}}, v{{\[[0-9]:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @trunc_store_v8i64_v8i16(ptr addrspace(1) %out, <8 x i64> %in) {
entry:
  %trunc = trunc <8 x i64> %in to < 8 x i16>
  store <8 x i16> %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_store_v8i64_v8i32:
; GCN: global_store_dwordx4 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} offset:16
; GCN: global_store_dwordx4 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @trunc_store_v8i64_v8i32(ptr addrspace(1) %out, <8 x i64> %in) {
entry:
  %trunc = trunc <8 x i64> %in to <8 x i32>
  store <8 x i32> %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_store_v16i64_v16i32:
; GCN: global_store_dwordx4 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} offset:48
; GCN: global_store_dwordx4 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} offset:32
; GCN: global_store_dwordx4 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} offset:16
; GCN: global_store_dwordx4 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @trunc_store_v16i64_v16i32(ptr addrspace(1) %out, <16 x i64> %in) {
entry:
  %trunc = trunc <16 x i64> %in to <16 x i32>
  store <16 x i32> %trunc, ptr addrspace(1) %out
  ret void
}
