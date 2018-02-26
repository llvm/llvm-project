; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}local_store_i56:
; GCN-DAG: ds_write_b8 v0, v{{[0-9]+}} offset:6
; GCN-DAG: ds_write_b16 v0, v{{[0-9]+}} offset:4
; GCN-DAG: ds_write_b32 v0, v{{[0-9]+$}}
define void @local_store_i56(i56 addrspace(3)* %ptr, i56 %arg) #0 {
  store i56 %arg, i56 addrspace(3)* %ptr, align 8
  ret void
}

; GCN-LABEL: {{^}}local_store_i55:
; GCN-DAG: ds_write_b8 v0, v{{[0-9]+}} offset:6
; GCN-DAG: ds_write_b16 v0, v{{[0-9]+}} offset:4
; GCN-DAG: ds_write_b32 v0, v{{[0-9]+$}}
define amdgpu_kernel void @local_store_i55(i55 addrspace(3)* %ptr, i55 %arg) #0 {
  store i55 %arg, i55 addrspace(3)* %ptr, align 8
  ret void
}

; GCN-LABEL: {{^}}local_store_i48:
; GCN-DAG: ds_write_b16 v0, v{{[0-9]+}} offset:4
; GCN-DAG: ds_write_b32 v0, v{{[0-9]+$}}
define amdgpu_kernel void @local_store_i48(i48 addrspace(3)* %ptr, i48 %arg) #0 {
  store i48 %arg, i48 addrspace(3)* %ptr, align 8
  ret void
}

; GCN-LABEL: {{^}}local_store_i65:
; GCN-DAG: ds_write_b8 v{{[0-9]+}}, v{{[0-9]+}} offset:8
; GCN-DAG: ds_write_b64
define amdgpu_kernel void @local_store_i65(i65 addrspace(3)* %ptr, i65 %arg) #0 {
  store i65 %arg, i65 addrspace(3)* %ptr, align 8
  ret void
}

; GCN-LABEL: {{^}}local_store_i13:
; GCN: v_and_b32_e32 [[TRUNC:v[0-9]+]], 0x1fff, v1
; GCN: ds_write_b16 v0, [[TRUNC]]
define void @local_store_i13(i13 addrspace(3)* %ptr, i13 %arg) #0 {
  store i13 %arg, i13 addrspace(3)* %ptr, align 8
  ret void
}

; GCN-LABEL: {{^}}local_store_i17:
; GCN: ds_write_b16 v0
; CIVI: ds_write_b8 v0, v{{[0-9]+}} offset:2
; GFX9: ds_write_b8_d16_hi v0, v{{[0-9]+}} offset:2
define void @local_store_i17(i17 addrspace(3)* %ptr, i17 %arg) #0 {
  store i17 %arg, i17 addrspace(3)* %ptr, align 8
  ret void
}

attributes #0 = { nounwind }
