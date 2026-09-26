; RUN: llc -O0 -global-isel=false -mtriple=amdgpu9.0a-amd-amdhsa %s -o - | FileCheck %s --check-prefix=GFX90A
; RUN: llc -O0 -global-isel=false -mtriple=amdgpu12.01-amd-amdhsa %s -o - | FileCheck %s --check-prefix=GFX12

define amdgpu_kernel void @store_as4_v4i64(ptr addrspace(4) %dst, <4 x i64> %value) {
; GFX90A-LABEL: store_as4_v4i64:
; GFX90A-COUNT-2: global_store_dwordx4
; GFX12-LABEL: store_as4_v4i64:
; GFX12-COUNT-2: global_store_b128
  store <4 x i64> %value, ptr addrspace(4) %dst, align 32
  ret void
}
