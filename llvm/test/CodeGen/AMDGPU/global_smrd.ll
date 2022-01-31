; RUN: llc -mtriple amdgcn--amdhsa -mcpu=fiji -amdgpu-scalarize-global-loads=true -verify-machineinstrs < %s | FileCheck %s

; uniform loads
; CHECK-LABEL: @uniform_load
; CHECK: s_load_dwordx4
; CHECK-NOT: flat_load_dword

define amdgpu_kernel void @uniform_load(float addrspace(1)* %arg, [8 x i32], float addrspace(1)* %arg1) {
bb:
  %tmp2 = load float, float addrspace(1)* %arg, align 4, !tbaa !8
  %tmp3 = fadd float %tmp2, 0.000000e+00
  %tmp4 = getelementptr inbounds float, float addrspace(1)* %arg, i64 1
  %tmp5 = load float, float addrspace(1)* %tmp4, align 4, !tbaa !8
  %tmp6 = fadd float %tmp3, %tmp5
  %tmp7 = getelementptr inbounds float, float addrspace(1)* %arg, i64 2
  %tmp8 = load float, float addrspace(1)* %tmp7, align 4, !tbaa !8
  %tmp9 = fadd float %tmp6, %tmp8
  %tmp10 = getelementptr inbounds float, float addrspace(1)* %arg, i64 3
  %tmp11 = load float, float addrspace(1)* %tmp10, align 4, !tbaa !8
  %tmp12 = fadd float %tmp9, %tmp11
  %tmp13 = getelementptr inbounds float, float addrspace(1)* %arg1
  store float %tmp12, float addrspace(1)* %tmp13, align 4, !tbaa !8
  ret void
}

; uniform loads before and after an aliasing store
; FIXME: The second load should not be converted to an SMEM load!
; CHECK-LABEL: @uniform_load_store_load
; CHECK: s_load_dwordx4
; CHECK: s_load_dword
; CHECK: flat_store_dword
; CHECK: s_load_dword
; CHECK: flat_store_dword

define amdgpu_kernel void @uniform_load_store_load(float addrspace(1)* %arg0, float addrspace(1)* %arg1) {
bb:
  %tmp2 = load float, float addrspace(1)* %arg0, !tbaa !8
  store float %tmp2, float addrspace(1)* %arg1, !tbaa !8
  %tmp3 = load float, float addrspace(1)* %arg0, !tbaa !8
  store float %tmp3, float addrspace(1)* %arg1, !tbaa !8
  ret void
}

; non-uniform loads
; CHECK-LABEL: @non-uniform_load
; CHECK: flat_load_dword
; CHECK-NOT: s_load_dwordx4

define amdgpu_kernel void @non-uniform_load(float addrspace(1)* %arg, [8 x i32], float addrspace(1)* %arg1) #0 {
bb:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp2 = getelementptr inbounds float, float addrspace(1)* %arg, i32 %tmp
  %tmp3 = load float, float addrspace(1)* %tmp2, align 4, !tbaa !8
  %tmp4 = fadd float %tmp3, 0.000000e+00
  %tmp5 = add i32 %tmp, 1
  %tmp6 = getelementptr inbounds float, float addrspace(1)* %arg, i32 %tmp5
  %tmp7 = load float, float addrspace(1)* %tmp6, align 4, !tbaa !8
  %tmp8 = fadd float %tmp4, %tmp7
  %tmp9 = add i32 %tmp, 2
  %tmp10 = getelementptr inbounds float, float addrspace(1)* %arg, i32 %tmp9
  %tmp11 = load float, float addrspace(1)* %tmp10, align 4, !tbaa !8
  %tmp12 = fadd float %tmp8, %tmp11
  %tmp13 = add i32 %tmp, 3
  %tmp14 = getelementptr inbounds float, float addrspace(1)* %arg, i32 %tmp13
  %tmp15 = load float, float addrspace(1)* %tmp14, align 4, !tbaa !8
  %tmp16 = fadd float %tmp12, %tmp15
  %tmp17 = getelementptr inbounds float, float addrspace(1)* %arg1, i32 %tmp
  store float %tmp16, float addrspace(1)* %tmp17, align 4, !tbaa !8
  ret void
}


; uniform load dominated by no-alias store - scalarize
; CHECK-LABEL: @no_memdep_alias_arg
; CHECK: s_load_dwordx2 s{{\[}}[[IN_LO:[0-9]+]]:[[IN_HI:[0-9]+]]], s[4:5], 0x0
; CHECK: s_load_dword [[SVAL:s[0-9]+]], s{{\[}}[[IN_LO]]:[[IN_HI]]], 0x0
; CHECK: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[SVAL]]
; CHECK: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[VVAL]]

define amdgpu_kernel void @no_memdep_alias_arg(i32 addrspace(1)* noalias %in, [8 x i32], i32 addrspace(1)* %out0, [8 x i32], i32 addrspace(1)* %out1) {
  store i32 0, i32 addrspace(1)* %out0
  %val = load i32, i32 addrspace(1)* %in
  store i32 %val, i32 addrspace(1)* %out1
  ret void
}

; uniform load dominated by alias store - vector
; CHECK-LABEL: {{^}}memdep:
; CHECK: flat_store_dword
; CHECK: flat_load_dword [[VVAL:v[0-9]+]]
; CHECK: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[VVAL]]
define amdgpu_kernel void @memdep(i32 addrspace(1)* %in, [8 x i32], i32 addrspace(1)* %out0, [8 x i32], i32 addrspace(1)* %out1) {
  store i32 0, i32 addrspace(1)* %out0
  %val = load i32, i32 addrspace(1)* %in
  store i32 %val, i32 addrspace(1)* %out1
  ret void
}

; uniform load from global array
; CHECK-LABEL:  @global_array
; CHECK: s_getpc_b64 [[GET_PC:s\[[0-9]+:[0-9]+\]]]
; CHECK-DAG: s_load_dwordx2 [[A_ADDR:s\[[0-9]+:[0-9]+\]]], [[GET_PC]], 0x0
; CHECK-DAG: s_load_dwordx2 [[A_ADDR1:s\[[0-9]+:[0-9]+\]]], [[A_ADDR]], 0x0
; CHECK-DAG: s_load_dwordx2 [[OUT:s\[[0-9]+:[0-9]+\]]], s[4:5], 0x0
; CHECK-DAG: s_load_dword [[SVAL:s[0-9]+]], [[A_ADDR1]], 0x0
; CHECK: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[SVAL]]
; CHECK: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[VVAL]]
@A = common local_unnamed_addr addrspace(1) global i32 addrspace(1)* null, align 4

define amdgpu_kernel void @global_array(i32 addrspace(1)* nocapture %out) {
entry:
  %load0 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* @A, align 4
  %load1 = load i32, i32 addrspace(1)* %load0, align 4
  store i32 %load1, i32 addrspace(1)* %out, align 4
  ret void
}


; uniform load from global array dominated by alias store
; CHECK-LABEL:  @global_array_alias_store
; CHECK: flat_store_dword
; CHECK: v_mov_b32_e32 v[[ADDR_LO:[0-9]+]], s{{[0-9]+}}
; CHECK: v_mov_b32_e32 v[[ADDR_HI:[0-9]+]], s{{[0-9]+}}
; CHECK: flat_load_dwordx2 [[A_ADDR:v\[[0-9]+:[0-9]+\]]], v{{\[}}[[ADDR_LO]]:[[ADDR_HI]]{{\]}}
; CHECK: flat_load_dword [[VVAL:v[0-9]+]], [[A_ADDR]]
; CHECK: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[VVAL]]
define amdgpu_kernel void @global_array_alias_store(i32 addrspace(1)* nocapture %out, [8 x i32], i32 %n) {
entry:
  %gep = getelementptr i32, i32 addrspace(1) * %out, i32 %n
  store i32 12, i32 addrspace(1) * %gep
  %load0 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* @A, align 4
  %load1 = load i32, i32 addrspace(1)* %load0, align 4
  store i32 %load1, i32 addrspace(1)* %out, align 4
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #1 = { nounwind readnone }

!8 = !{!9, !9, i64 0}
!9 = !{!"float", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
