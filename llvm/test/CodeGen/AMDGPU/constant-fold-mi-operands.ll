; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}fold_mi_v_and_0:
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], 0{{$}}
; GCN-NOT: [[RESULT]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @fold_mi_v_and_0(i32 addrspace(1)* %out) {
  %x = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %and = and i32 %size, %x
  store i32 %and, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fold_mi_s_and_0:
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], 0{{$}}
; GCN-NOT: [[RESULT]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @fold_mi_s_and_0(i32 addrspace(1)* %out, i32 %x) #0 {
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %and = and i32 %size, %x
  store i32 %and, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fold_mi_v_or_0:
; GCN: v_mbcnt_lo_u32_b32_e64 [[RESULT:v[0-9]+]]
; GCN-NOT: [[RESULT]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @fold_mi_v_or_0(i32 addrspace(1)* %out) {
  %x = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %or = or i32 %size, %x
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fold_mi_s_or_0:
; GCN: s_load_dword [[SVAL:s[0-9]+]]
; GCN-NOT: [[SVAL]]
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[SVAL]]
; GCN-NOT: [[VVAL]]
; GCN: buffer_store_dword [[VVAL]]
define amdgpu_kernel void @fold_mi_s_or_0(i32 addrspace(1)* %out, i32 %x) #0 {
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %or = or i32 %size, %x
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fold_mi_v_xor_0:
; GCN: v_mbcnt_lo_u32_b32_e64 [[RESULT:v[0-9]+]]
; GCN-NOT: [[RESULT]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @fold_mi_v_xor_0(i32 addrspace(1)* %out) {
  %x = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %xor = xor i32 %size, %x
  store i32 %xor, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fold_mi_s_xor_0:
; GCN: s_load_dword [[SVAL:s[0-9]+]]
; GCN-NOT: [[SVAL]]
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[SVAL]]
; GCN-NOT: [[VVAL]]
; GCN: buffer_store_dword [[VVAL]]
define amdgpu_kernel void @fold_mi_s_xor_0(i32 addrspace(1)* %out, i32 %x) #0 {
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %xor = xor i32 %size, %x
  store i32 %xor, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fold_mi_s_not_0:
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], -1{{$}}
; GCN-NOT: [[RESULT]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @fold_mi_s_not_0(i32 addrspace(1)* %out, i32 %x) #0 {
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %xor = xor i32 %size, -1
  store i32 %xor, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fold_mi_v_not_0:
; GCN: v_bcnt_u32_b32_e64 v[[RESULT_LO:[0-9]+]], v{{[0-9]+}}, 0{{$}}
; GCN: v_bcnt_u32_b32_e{{[0-9]+}} v[[RESULT_LO:[0-9]+]], v{{[0-9]+}}, v[[RESULT_LO]]{{$}}
; GCN-NEXT: v_not_b32_e32 v[[RESULT_LO]]
; GCN-NEXT: v_mov_b32_e32 v[[RESULT_HI:[0-9]+]], -1{{$}}
; GCN-NEXT: buffer_store_dwordx2 v{{\[}}[[RESULT_LO]]:[[RESULT_HI]]{{\]}}
define amdgpu_kernel void @fold_mi_v_not_0(i64 addrspace(1)* %out) {
  %vreg = load volatile i64, i64 addrspace(1)* undef
  %ctpop = call i64 @llvm.ctpop.i64(i64 %vreg)
  %xor = xor i64 %ctpop, -1
  store i64 %xor, i64 addrspace(1)* %out
  ret void
}

; The neg1 appears after folding the not 0
; GCN-LABEL: {{^}}fold_mi_or_neg1:
; GCN: buffer_load_dwordx2
; GCN: buffer_load_dwordx2 v{{\[}}[[VREG1_LO:[0-9]+]]:[[VREG1_HI:[0-9]+]]{{\]}}

; GCN: v_bcnt_u32_b32_e64 v[[RESULT_LO:[0-9]+]], v{{[0-9]+}}, 0{{$}}
; GCN: v_bcnt_u32_b32_e{{[0-9]+}} v[[RESULT_LO:[0-9]+]], v{{[0-9]+}}, v[[RESULT_LO]]{{$}}
; GCN-DAG: v_not_b32_e32 v[[RESULT_LO]], v[[RESULT_LO]]
; GCN-DAG: v_or_b32_e32 v[[RESULT_LO]], v[[VREG1_LO]], v[[RESULT_LO]]
; GCN-DAG: v_mov_b32_e32 v[[RESULT_HI:[0-9]+]], v[[VREG1_HI]]
; GCN: buffer_store_dwordx2 v{{\[}}[[RESULT_LO]]:[[RESULT_HI]]{{\]}}
define amdgpu_kernel void @fold_mi_or_neg1(i64 addrspace(1)* %out) {
  %vreg0 = load volatile i64, i64 addrspace(1)* undef
  %vreg1 = load volatile i64, i64 addrspace(1)* undef
  %ctpop = call i64 @llvm.ctpop.i64(i64 %vreg0)
  %xor = xor i64 %ctpop, -1
  %or = or i64 %xor, %vreg1
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fold_mi_and_neg1:
; GCN: v_bcnt_u32_b32
; GCN: v_bcnt_u32_b32
; GCN: v_not_b32
; GCN: v_and_b32
; GCN-NOT: v_and_b32
define amdgpu_kernel void @fold_mi_and_neg1(i64 addrspace(1)* %out) {
  %vreg0 = load volatile i64, i64 addrspace(1)* undef
  %vreg1 = load volatile i64, i64 addrspace(1)* undef
  %ctpop = call i64 @llvm.ctpop.i64(i64 %vreg0)
  %xor = xor i64 %ctpop, -1
  %and = and i64 %xor, %vreg1
  store i64 %and, i64 addrspace(1)* %out
  ret void
}

declare i64 @llvm.ctpop.i64(i64) #1
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #1
declare i32 @llvm.amdgcn.groupstaticsize() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
