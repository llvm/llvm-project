; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}atomic_sub_local:
; R600: LDS_SUB *
; SI: ds_sub_u32
define void @atomic_sub_local(i32 addrspace(3)* %local) {
   %unused = atomicrmw volatile sub i32 addrspace(3)* %local, i32 5 seq_cst
   ret void
}

; FUNC-LABEL: {{^}}atomic_sub_local_const_offset:
; R600: LDS_SUB *
; SI: ds_sub_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
define void @atomic_sub_local_const_offset(i32 addrspace(3)* %local) {
  %gep = getelementptr i32, i32 addrspace(3)* %local, i32 4
  %val = atomicrmw volatile sub i32 addrspace(3)* %gep, i32 5 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_ret_local:
; R600: LDS_SUB_RET *
; SI: ds_sub_rtn_u32
define void @atomic_sub_ret_local(i32 addrspace(1)* %out, i32 addrspace(3)* %local) {
  %val = atomicrmw volatile sub i32 addrspace(3)* %local, i32 5 seq_cst
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_ret_local_const_offset:
; R600: LDS_SUB_RET *
; SI: ds_sub_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:20
define void @atomic_sub_ret_local_const_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %local) {
  %gep = getelementptr i32, i32 addrspace(3)* %local, i32 5
  %val = atomicrmw volatile sub i32 addrspace(3)* %gep, i32 5 seq_cst
  store i32 %val, i32 addrspace(1)* %out
  ret void
}
