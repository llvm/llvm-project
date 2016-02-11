; RUN: llc -march=amdgcn -mcpu=SI -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=SI-ALLOCA -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=SI -mattr=+promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=SI-PROMOTE -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=SI-ALLOCA -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=+promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=SI-PROMOTE -check-prefix=SI %s

declare void @llvm.amdgcn.s.barrier() #1

; SI-LABEL: {{^}}private_access_f64_alloca:

; SI-ALLOCA: buffer_store_dwordx2
; SI-ALLOCA: buffer_load_dwordx2

; SI-PROMOTE: ds_write_b64
; SI-PROMOTE: ds_read_b64
define void @private_access_f64_alloca(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in, i32 %b) #1 {
  %val = load double, double addrspace(1)* %in, align 8
  %array = alloca double, i32 16, align 8
  %ptr = getelementptr inbounds double, double* %array, i32 %b
  store double %val, double* %ptr, align 8
  call void @llvm.amdgcn.s.barrier()
  %result = load double, double* %ptr, align 8
  store double %result, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}private_access_v2f64_alloca:

; SI-ALLOCA: buffer_store_dwordx4
; SI-ALLOCA: buffer_load_dwordx4

; SI-PROMOTE: ds_write_b64
; SI-PROMOTE: ds_write_b64
; SI-PROMOTE: ds_read_b64
; SI-PROMOTE: ds_read_b64
define void @private_access_v2f64_alloca(<2 x double> addrspace(1)* noalias %out, <2 x double> addrspace(1)* noalias %in, i32 %b) #1 {
  %val = load <2 x double>, <2 x double> addrspace(1)* %in, align 16
  %array = alloca <2 x double>, i32 16, align 16
  %ptr = getelementptr inbounds <2 x double>, <2 x double>* %array, i32 %b
  store <2 x double> %val, <2 x double>* %ptr, align 16
  call void @llvm.amdgcn.s.barrier()
  %result = load <2 x double>, <2 x double>* %ptr, align 16
  store <2 x double> %result, <2 x double> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: {{^}}private_access_i64_alloca:

; SI-ALLOCA: buffer_store_dwordx2
; SI-ALLOCA: buffer_load_dwordx2

; SI-PROMOTE: ds_write_b64
; SI-PROMOTE: ds_read_b64
define void @private_access_i64_alloca(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in, i32 %b) #1 {
  %val = load i64, i64 addrspace(1)* %in, align 8
  %array = alloca i64, i32 16, align 8
  %ptr = getelementptr inbounds i64, i64* %array, i32 %b
  store i64 %val, i64* %ptr, align 8
  call void @llvm.amdgcn.s.barrier()
  %result = load i64, i64* %ptr, align 8
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}private_access_v2i64_alloca:

; SI-ALLOCA: buffer_store_dwordx4
; SI-ALLOCA: buffer_load_dwordx4

; SI-PROMOTE: ds_write_b64
; SI-PROMOTE: ds_write_b64
; SI-PROMOTE: ds_read_b64
; SI-PROMOTE: ds_read_b64
define void @private_access_v2i64_alloca(<2 x i64> addrspace(1)* noalias %out, <2 x i64> addrspace(1)* noalias %in, i32 %b) #1 {
  %val = load <2 x i64>, <2 x i64> addrspace(1)* %in, align 16
  %array = alloca <2 x i64>, i32 16, align 16
  %ptr = getelementptr inbounds <2 x i64>, <2 x i64>* %array, i32 %b
  store <2 x i64> %val, <2 x i64>* %ptr, align 16
  call void @llvm.amdgcn.s.barrier()
  %result = load <2 x i64>, <2 x i64>* %ptr, align 16
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out, align 16
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind convergent }
