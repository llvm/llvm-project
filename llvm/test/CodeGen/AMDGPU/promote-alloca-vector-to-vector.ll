; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: opt -S -mtriple=amdgcn-- -data-layout=A5 -mcpu=fiji -passes=sroa,amdgpu-promote-alloca < %s | FileCheck -check-prefix=OPT %s

; GCN-LABEL: {{^}}float4_alloca_store4:
; OPT-LABEL: define amdgpu_kernel void @float4_alloca_store4

; GCN-NOT: buffer_
; GCN: v_cndmask_b32
; GCN: v_cndmask_b32
; GCN: v_cndmask_b32_e32 [[RES:v[0-9]+]], 4.0,
; GCN: store_dword v{{.+}}, [[RES]]

; OPT:  %0 = extractelement <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>, i32 %sel2
; OPT:  store float %0, ptr addrspace(1) %out, align 4

define amdgpu_kernel void @float4_alloca_store4(ptr addrspace(1) %out, ptr addrspace(3) %dummy_lds) {
entry:
  %alloca = alloca <4 x float>, align 16, addrspace(5)
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %c1 = icmp uge i32 %x, 3
  %c2 = icmp uge i32 %y, 3
  %sel1 = select i1 %c1, i32 1, i32 2
  %sel2 = select i1 %c2, i32 0, i32 %sel1
  %gep = getelementptr inbounds <4 x float>, ptr addrspace(5) %alloca, i32 0, i32 %sel2
  store <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, ptr addrspace(5) %alloca, align 4
  %load = load float, ptr addrspace(5) %gep, align 4
  store float %load, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}float4_alloca_load4:
; OPT-LABEL: define amdgpu_kernel void @float4_alloca_load4

; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN-NOT: v_cmp_
; GCN-NOT: v_cndmask_
; GCN:     v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN:     v_mov_b32_e32 v{{[0-9]+}}, [[ONE]]
; GCN:     v_mov_b32_e32 v{{[0-9]+}}, [[ONE]]
; GCN:     v_mov_b32_e32 v{{[0-9]+}}, [[ONE]]
; GCN:     store_dwordx4 v{{.+}},

; OPT: %0 = insertelement <4 x float> undef, float 1.000000e+00, i32 %sel2
; OPT: store <4 x float> %0, ptr addrspace(1) %out, align 4

define amdgpu_kernel void @float4_alloca_load4(ptr addrspace(1) %out, ptr addrspace(3) %dummy_lds) {
entry:
  %alloca = alloca <4 x float>, align 16, addrspace(5)
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %c1 = icmp uge i32 %x, 3
  %c2 = icmp uge i32 %y, 3
  %sel1 = select i1 %c1, i32 1, i32 2
  %sel2 = select i1 %c2, i32 0, i32 %sel1
  %gep = getelementptr inbounds <4 x float>, ptr addrspace(5) %alloca, i32 0, i32 %sel2
  store float 1.0, ptr addrspace(5) %gep, align 4
  %load = load <4 x float>, ptr addrspace(5) %alloca, align 4
  store <4 x float> %load, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}half4_alloca_store4:
; OPT-LABEL: define amdgpu_kernel void @half4_alloca_store4

; GCN-NOT: buffer_
; GCN-DAG: s_mov_b32 s[[SH:[0-9]+]], 0x44004200
; GCN-DAG: s_mov_b32 s[[SL:[0-9]+]], 0x40003c00
; GCN:     v_lshrrev_b64 v[{{[0-9:]+}}], v{{[0-9]+}}, s[[[SL]]:[[SH]]]

; OPT: %0 = extractelement <4 x half> <half 0xH3C00, half 0xH4000, half 0xH4200, half 0xH4400>, i32 %sel2
; OPT: store half %0, ptr addrspace(1) %out, align 2

define amdgpu_kernel void @half4_alloca_store4(ptr addrspace(1) %out, ptr addrspace(3) %dummy_lds) {
entry:
  %alloca = alloca <4 x half>, align 16, addrspace(5)
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %c1 = icmp uge i32 %x, 3
  %c2 = icmp uge i32 %y, 3
  %sel1 = select i1 %c1, i32 1, i32 2
  %sel2 = select i1 %c2, i32 0, i32 %sel1
  %gep = getelementptr inbounds <4 x half>, ptr addrspace(5) %alloca, i32 0, i32 %sel2
  store <4 x half> <half 1.0, half 2.0, half 3.0, half 4.0>, ptr addrspace(5) %alloca, align 2
  %load = load half, ptr addrspace(5) %gep, align 2
  store half %load, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}half4_alloca_load4:
; OPT-LABEL: define amdgpu_kernel void @half4_alloca_load4

; GCN-NOT: buffer_
; GCN:     s_mov_b64 s[{{[0-9:]+}}], 0xffff

; OPT: %0 = insertelement <4 x half> undef, half 0xH3C00, i32 %sel2
; OPT: store <4 x half> %0, ptr addrspace(1) %out, align 2

define amdgpu_kernel void @half4_alloca_load4(ptr addrspace(1) %out, ptr addrspace(3) %dummy_lds) {
entry:
  %alloca = alloca <4 x half>, align 16, addrspace(5)
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %c1 = icmp uge i32 %x, 3
  %c2 = icmp uge i32 %y, 3
  %sel1 = select i1 %c1, i32 1, i32 2
  %sel2 = select i1 %c2, i32 0, i32 %sel1
  %gep = getelementptr inbounds <4 x half>, ptr addrspace(5) %alloca, i32 0, i32 %sel2
  store half 1.0, ptr addrspace(5) %gep, align 4
  %load = load <4 x half>, ptr addrspace(5) %alloca, align 2
  store <4 x half> %load, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}short4_alloca_store4:
; OPT-LABEL: define amdgpu_kernel void @short4_alloca_store4

; GCN-NOT: buffer_
; GCN-DAG: s_mov_b32 s[[SH:[0-9]+]], 0x40003
; GCN-DAG: s_mov_b32 s[[SL:[0-9]+]], 0x20001
; GCN:     v_lshrrev_b64 v[{{[0-9:]+}}], v{{[0-9]+}}, s[[[SL]]:[[SH]]]

; OPT: %0 = extractelement <4 x i16> <i16 1, i16 2, i16 3, i16 4>, i32 %sel2
; OPT: store i16 %0, ptr addrspace(1) %out, align 2

define amdgpu_kernel void @short4_alloca_store4(ptr addrspace(1) %out, ptr addrspace(3) %dummy_lds) {
entry:
  %alloca = alloca <4 x i16>, align 16, addrspace(5)
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %c1 = icmp uge i32 %x, 3
  %c2 = icmp uge i32 %y, 3
  %sel1 = select i1 %c1, i32 1, i32 2
  %sel2 = select i1 %c2, i32 0, i32 %sel1
  %gep = getelementptr inbounds <4 x i16>, ptr addrspace(5) %alloca, i32 0, i32 %sel2
  store <4 x i16> <i16 1, i16 2, i16 3, i16 4>, ptr addrspace(5) %alloca, align 2
  %load = load i16, ptr addrspace(5) %gep, align 2
  store i16 %load, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}short4_alloca_load4:
; OPT-LABEL: define amdgpu_kernel void @short4_alloca_load4

; GCN-NOT: buffer_
; GCN:     s_mov_b64 s[{{[0-9:]+}}], 0xffff

; OPT: %0 = insertelement <4 x i16> undef, i16 1, i32 %sel2
; OPT: store <4 x i16> %0, ptr addrspace(1) %out, align 2

define amdgpu_kernel void @short4_alloca_load4(ptr addrspace(1) %out, ptr addrspace(3) %dummy_lds) {
entry:
  %alloca = alloca <4 x i16>, align 16, addrspace(5)
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %c1 = icmp uge i32 %x, 3
  %c2 = icmp uge i32 %y, 3
  %sel1 = select i1 %c1, i32 1, i32 2
  %sel2 = select i1 %c2, i32 0, i32 %sel1
  %gep = getelementptr inbounds <4 x i16>, ptr addrspace(5) %alloca, i32 0, i32 %sel2
  store i16 1, ptr addrspace(5) %gep, align 4
  %load = load <4 x i16>, ptr addrspace(5) %alloca, align 2
  store <4 x i16> %load, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}ptr_alloca_bitcast:
; OPT-LABEL: define i64 @ptr_alloca_bitcast

; GCN-NOT: buffer_
; GCN: v_mov_b32_e32 v1, 0

; OPT: ret i64 undef

define i64 @ptr_alloca_bitcast() {
entry:
  %private_iptr = alloca <2 x i32>, align 8, addrspace(5)
  %tmp1 = load i64, ptr addrspace(5) %private_iptr, align 8
  ret i64 %tmp1
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workitem.id.y()
