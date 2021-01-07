; RUN: llc -march=amdgcn -mattr=-promote-alloca,+max-private-element-size-16 -verify-machineinstrs < %s | FileCheck --check-prefixes=SI-ALLOCA16,SI %s
; RUN: llc -march=amdgcn -mattr=-promote-alloca,+max-private-element-size-4 -verify-machineinstrs < %s | FileCheck --check-prefixes=SI-ALLOCA4,SI %s
; RUN: llc -march=amdgcn -mattr=+promote-alloca -disable-promote-alloca-to-vector -verify-machineinstrs < %s | FileCheck --check-prefixes=SI-PROMOTE,SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -mattr=-promote-alloca,+max-private-element-size-16 -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -mattr=+promote-alloca -disable-promote-alloca-to-vector -verify-machineinstrs < %s | FileCheck --check-prefixes=CI-PROMOTE,SI %s

declare void @llvm.amdgcn.s.barrier() #0

; SI-LABEL: {{^}}private_access_f64_alloca:

; SI-ALLOCA16: buffer_store_dwordx2
; SI-ALLOCA16: buffer_load_dwordx2

; SI-ALLOCA4: buffer_store_dword v
; SI-ALLOCA4: buffer_store_dword v
; SI-ALLOCA4: buffer_load_dword v
; SI-ALLOCA4: buffer_load_dword v

; SI-PROMOTE: ds_write_b64
; SI-PROMOTE: ds_read_b64
; CI-PROMOTE: ds_write_b64
; CI-PROMOTE: ds_read_b64
define amdgpu_kernel void @private_access_f64_alloca(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in, i32 %b) #1 {
  %val = load double, double addrspace(1)* %in, align 8
  %array = alloca [8 x double], align 8, addrspace(5)
  %ptr = getelementptr inbounds [8 x double], [8 x double] addrspace(5)* %array, i32 0, i32 %b
  store double %val, double addrspace(5)* %ptr, align 8
  call void @llvm.amdgcn.s.barrier()
  %result = load double, double addrspace(5)* %ptr, align 8
  store double %result, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}private_access_v2f64_alloca:

; SI-ALLOCA16: buffer_store_dwordx4
; SI-ALLOCA16: buffer_load_dwordx4

; SI-ALLOCA4: buffer_store_dword v
; SI-ALLOCA4: buffer_store_dword v
; SI-ALLOCA4: buffer_store_dword v
; SI-ALLOCA4: buffer_store_dword v
; SI-ALLOCA4: buffer_load_dword v
; SI-ALLOCA4: buffer_load_dword v
; SI-ALLOCA4: buffer_load_dword v
; SI-ALLOCA4: buffer_load_dword v

; SI-PROMOTE: ds_write_b64
; SI-PROMOTE: ds_write_b64
; SI-PROMOTE: ds_read_b64
; SI-PROMOTE: ds_read_b64
; CI-PROMOTE: ds_write_b128
; CI-PROMOTE: ds_read_b128
define amdgpu_kernel void @private_access_v2f64_alloca(<2 x double> addrspace(1)* noalias %out, <2 x double> addrspace(1)* noalias %in, i32 %b) #1 {
  %val = load <2 x double>, <2 x double> addrspace(1)* %in, align 16
  %array = alloca [4 x <2 x double>], align 16, addrspace(5)
  %ptr = getelementptr inbounds [4 x <2 x double>], [4 x <2 x double>] addrspace(5)* %array, i32 0, i32 %b
  store <2 x double> %val, <2 x double> addrspace(5)* %ptr, align 16
  call void @llvm.amdgcn.s.barrier()
  %result = load <2 x double>, <2 x double> addrspace(5)* %ptr, align 16
  store <2 x double> %result, <2 x double> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: {{^}}private_access_i64_alloca:

; SI-ALLOCA16: buffer_store_dwordx2
; SI-ALLOCA16: buffer_load_dwordx2

; SI-ALLOCA4: buffer_store_dword v
; SI-ALLOCA4: buffer_store_dword v
; SI-ALLOCA4: buffer_load_dword v
; SI-ALLOCA4: buffer_load_dword v


; SI-PROMOTE: ds_write_b64
; SI-PROMOTE: ds_read_b64
; CI-PROMOTE: ds_write_b64
; CI-PROMOTE: ds_read_b64
define amdgpu_kernel void @private_access_i64_alloca(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in, i32 %b) #1 {
  %val = load i64, i64 addrspace(1)* %in, align 8
  %array = alloca [8 x i64], align 8, addrspace(5)
  %ptr = getelementptr inbounds [8 x i64], [8 x i64] addrspace(5)* %array, i32 0, i32 %b
  store i64 %val, i64 addrspace(5)* %ptr, align 8
  call void @llvm.amdgcn.s.barrier()
  %result = load i64, i64 addrspace(5)* %ptr, align 8
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}private_access_v2i64_alloca:

; SI-ALLOCA16: buffer_store_dwordx4
; SI-ALLOCA16: buffer_load_dwordx4

; SI-ALLOCA4: buffer_store_dword v
; SI-ALLOCA4: buffer_store_dword v
; SI-ALLOCA4: buffer_store_dword v
; SI-ALLOCA4: buffer_store_dword v

; SI-ALLOCA4: buffer_load_dword v
; SI-ALLOCA4: buffer_load_dword v
; SI-ALLOCA4: buffer_load_dword v
; SI-ALLOCA4: buffer_load_dword v

; SI-PROMOTE: ds_write_b64
; SI-PROMOTE: ds_write_b64
; SI-PROMOTE: ds_read_b64
; SI-PROMOTE: ds_read_b64
; CI-PROMOTE: ds_write_b128
; CI-PROMOTE: ds_read_b128
define amdgpu_kernel void @private_access_v2i64_alloca(<2 x i64> addrspace(1)* noalias %out, <2 x i64> addrspace(1)* noalias %in, i32 %b) #1 {
  %val = load <2 x i64>, <2 x i64> addrspace(1)* %in, align 16
  %array = alloca [4 x <2 x i64>], align 16, addrspace(5)
  %ptr = getelementptr inbounds [4 x <2 x i64>], [4 x <2 x i64>] addrspace(5)* %array, i32 0, i32 %b
  store <2 x i64> %val, <2 x i64> addrspace(5)* %ptr, align 16
  call void @llvm.amdgcn.s.barrier()
  %result = load <2 x i64>, <2 x i64> addrspace(5)* %ptr, align 16
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out, align 16
  ret void
}

attributes #0 = { convergent nounwind }
attributes #1 = { nounwind "amdgpu-waves-per-eu"="1,2" "amdgpu-flat-work-group-size"="64,128" }
